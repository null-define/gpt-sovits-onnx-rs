use async_stream::stream;
use futures::{Stream, StreamExt};
use hound::{WavReader, WavSpec};
use log::{debug, info};
use ndarray::{
    Array, Array2, ArrayBase, ArrayD, ArrayView2, Axis, IxDyn,
    OwnedRepr, concatenate, s,
};
use ort::{
    inputs,
    session::Session,
    value::TensorRef,
};
use std::time::SystemTime;
use std::{fs::File, path::Path};
use tokio::task::block_in_place;

mod cpu_info;
mod error;
mod logits_sampler;
mod onnx_builder;
mod text;

use onnx_builder::create_onnx_cpu_session;
pub use text::LangId;

use logits_sampler::Sampler;
use text::{TextProcessor, bert::BertModel, en::g2p_en::G2pEn, zh::g2pw::G2PW};

pub use error::GSVError;
pub use logits_sampler::{SamplingParams, SamplingParamsBuilder};

use crate::onnx_builder::BIG_CORES;

const T2S_DECODER_EOS: i64 = 1024;
const VOCAB_SIZE: usize = 1025;
const NUM_LAYERS: usize = 24;

type KvDType = f32;

#[derive(Clone)]
pub struct ReferenceData {
    ref_seq: Array2<i64>,
    ref_bert: Array2<f32>,
    ref_audio_32k: Array2<f32>,
    ssl_content: ArrayBase<OwnedRepr<f32>, IxDyn>,
}

pub struct TTSModel {
    text_processor: TextProcessor,
    sovits: Session,
    ssl: Session,
    t2s_encoder: Session,
    t2s_fs_decoder: Session,
    t2s_s_decoder: Session,
    ref_data: Option<ReferenceData>,
    num_layers: usize,
    output_spec: WavSpec,
}

// --- KV Cache Configuration ---
/// Initial size for the sequence length of the KV cache.
const INITIAL_CACHE_SIZE: usize = 2048;
/// How much to increment the KV cache size by when reallocating.
const CACHE_REALLOC_INCREMENT: usize = 1024;

impl TTSModel {
    /// create new tts instance
    /// bert_path, g2pw_path and g2p_en_path can be None
    /// if bert path is none, the speech speed in chinese may become worse
    /// if g2pw path is none, the chinese speech quality may be worse
    /// g2p_en is still experimental, english speak quality may not be better because of bugs
    pub fn new<P: AsRef<Path>>(
        sovits_path: P,
        ssl_path: P,
        t2s_encoder_path: P,
        t2s_fs_decoder_path: P,
        t2s_s_decoder_path: P,
        bert_path: Option<P>,
        g2pw_path: Option<P>,
        g2p_en_path: Option<P>,
    ) -> Result<Self, GSVError> {
        info!("Initializing TTSModel with ONNX sessions");
        info!("use cpu cores: {:?}", BIG_CORES.clone());

        // let create_session_with_profiling = |path: P| {
        //     Session::builder()?
        //         .with_execution_providers([CPUExecutionProvider::default()
        //             .with_arena_allocator(true)
        //             .build()])?
        //         .with_optimization_level(GraphOptimizationLevel::Level3)?
        //         .with_intra_threads(8)?
        //         .with_profiling("t2sd")?
        //         .commit_from_file(path)
        // };

        let output_spec = WavSpec {
            channels: 1,
            sample_rate: 32000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        Ok(TTSModel {
            text_processor: TextProcessor::new(
                G2PW::new(g2pw_path)?,
                G2pEn::new(g2p_en_path)?,
                BertModel::new(bert_path)?,
            )?,
            sovits: create_onnx_cpu_session(sovits_path)?,
            ssl: create_onnx_cpu_session(ssl_path)?,
            t2s_encoder: create_onnx_cpu_session(t2s_encoder_path)?,
            t2s_fs_decoder: create_onnx_cpu_session(t2s_fs_decoder_path)?,
            t2s_s_decoder: create_onnx_cpu_session(t2s_s_decoder_path)?,
            ref_data: None,
            num_layers: NUM_LAYERS,
            output_spec,
        })
    }

    fn run_async_in_context<F, T>(fut: F) -> Result<T, GSVError>
    where
        F: std::future::Future<Output = Result<T, GSVError>>,
    {
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => block_in_place(|| handle.block_on(fut)),
            Err(_) => {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(fut)
            }
        }
    }

    /// run reference with async fn
    ///
    /// `reference_audio_path` shall be input wav(16khz) path
    ///
    /// `ref_text` is input ref text
    ///
    /// `lang_id` can be LangId::Auto(Mandarin) or LangId::AutoYue（cantonese）
    ///
    pub async fn process_reference<P: AsRef<Path>>(
        &mut self,
        reference_audio_path: P,
        ref_text: &str,
        lang_id: LangId,
    ) -> Result<(), GSVError> {
        info!("Processing reference audio and text: {}", ref_text);
        let ref_text = ensure_punctuation(ref_text);
        let phones = self.text_processor.get_phone_and_bert(&ref_text, lang_id)?;
        let ref_seq: Vec<i64> = phones.iter().fold(Vec::new(), |mut seq, p| {
            seq.extend(p.1.clone());
            seq
        });

        let ref_bert: Vec<Array2<f32>> = phones.iter().map(|f| f.2.clone()).collect();
        // Concatenate along dimension 0
        let ref_bert = concatenate(
            Axis(0),
            &ref_bert.iter().map(|v| v.view()).collect::<Vec<_>>(),
        )
        .unwrap();

        let ref_seq = Array2::from_shape_vec((1, ref_seq.len()), ref_seq)?;
        let (ref_audio_16k, ref_audio_32k) = read_and_resample_audio(&reference_audio_path)?;
        let ssl_content = self.process_ssl(&ref_audio_16k)?;

        self.ref_data = Some(ReferenceData {
            ref_seq,
            ref_bert: ref_bert,
            ref_audio_32k,
            ssl_content,
        });

        Ok(())
    }

    fn process_ssl(
        &mut self,
        ref_audio_16k: &Array2<f32>,
    ) -> Result<ArrayBase<OwnedRepr<f32>, IxDyn>, GSVError> {
        let time = SystemTime::now();
        let ssl_output = self
            .ssl
            .run(inputs!["ref_audio_16k" => TensorRef::from_array_view(ref_audio_16k).unwrap()])?;
        debug!("SSL processing time: {:?}", time.elapsed()?);
        Ok(ssl_output["ssl_content"]
            .try_extract_array::<f32>()?
            .into_owned())
    }

    /// run reference
    ///
    /// `reference_audio_path` shall be input wav(16khz) path
    ///
    /// `ref_text` is input ref text
    ///
    /// `lang_id` can be LangId::Auto(Mandarin) or LangId::AutoYue（cantonese）
    ///
    pub fn process_reference_sync<P: AsRef<Path>>(
        &mut self,
        reference_audio_path: P,
        ref_text: &str,
        lang_id: LangId,
    ) -> Result<(), GSVError> {
        Self::run_async_in_context(self.process_reference(reference_audio_path, ref_text, lang_id))
    }

    /// Efficiently runs the streaming decoder loop with a pre-allocated, resizable KV cache.
    fn run_t2s_s_decoder_loop(
        &mut self,
        sampler: &mut Sampler,
        sampling_param: SamplingParams,
        mut y_vec: Vec<i64>,
        mut y_emb: ArrayBase<OwnedRepr<f32>, IxDyn>,
        x_example: ArrayBase<OwnedRepr<f32>, IxDyn>,
        mut k_caches: Vec<ArrayBase<OwnedRepr<KvDType>, IxDyn>>,
        mut v_caches: Vec<ArrayBase<OwnedRepr<KvDType>, IxDyn>>,
        prefix_len: usize,
        initial_valid_len: usize,
    ) -> Result<ArrayBase<OwnedRepr<i64>, IxDyn>, GSVError> {
        let mut idx = 1;
        let mut valid_len = initial_valid_len;
        y_vec.reserve(2048);

        loop {
            // --- 1. Prepare inputs using views of the valid cache portion ---
            // let time = SystemTime::now();
            let mut inputs = inputs![
                "iy" => TensorRef::from_array_view(unsafe {ArrayView2::from_shape_ptr((1, y_vec.len()), y_vec.as_ptr())}).unwrap(),
                "iy_emb" => TensorRef::from_array_view(&y_emb).unwrap(),
                "x_example" => TensorRef::from_array_view(&x_example).unwrap()
            ];

            for i in 0..self.num_layers {
                // Create a view of the valid part of the cache
                let k_view = k_caches[i].slice(s![0..valid_len, .., ..]);
                let v_view = v_caches[i].slice(s![0..valid_len, .., ..]);

                inputs.push((
                    format!("ik_cache_{}", i).into(),
                    TensorRef::from_array_view(k_view)?.into(),
                ));
                inputs.push((
                    format!("iv_cache_{}", i).into(),
                    TensorRef::from_array_view(v_view)?.into(),
                ));
            }
            // --- 2. Run the decoder model for one step ---
            let mut output = self.t2s_s_decoder.run(inputs)?;

            let mut logits = output["logits"].try_extract_array_mut::<f32>()?;
            let logits = logits.as_slice_mut().unwrap();

            y_vec.push(sampler.sample(
                logits,
                &y_vec,
                &sampling_param,
                if idx <= 10 {
                    &[0, T2S_DECODER_EOS]
                } else {
                    &[]
                },
            ));
            y_emb = output["y_emb"].try_extract_array::<f32>()?.into_owned();

            // --- 3. Check for reallocation and update caches ---
            let new_valid_len = valid_len + 1;

            // Check if we need to reallocate BEFORE writing to the new index.
            if new_valid_len > k_caches[0].shape()[0] {
                info!(
                    "Reallocating KV cache from {} to {}",
                    k_caches[0].shape()[0],
                    k_caches[0].shape()[0] + CACHE_REALLOC_INCREMENT
                );
                for i in 0..self.num_layers {
                    let old_k = &k_caches[i];
                    let old_v = &v_caches[i];

                    // Create new, larger arrays
                    let mut new_k_dims = old_k.raw_dim().clone();
                    new_k_dims[0] += CACHE_REALLOC_INCREMENT;
                    let mut new_v_dims = old_v.raw_dim().clone();
                    new_v_dims[0] += CACHE_REALLOC_INCREMENT;

                    let mut new_k = Array::zeros(new_k_dims);
                    let mut new_v = Array::zeros(new_v_dims);

                    // Copy existing valid data to the new arrays
                    new_k
                        .slice_mut(s![0..valid_len, ..])
                        .assign(&old_k.slice(s![0..valid_len, .., ..]));
                    new_v
                        .slice_mut(s![0..valid_len, ..])
                        .assign(&old_v.slice(s![0..valid_len, .., ..]));

                    // Replace the old caches with the new, larger ones
                    k_caches[i] = new_k;
                    v_caches[i] = new_v;
                }
            }

            // Update KV caches by pasting the newly generated slice of data
            for i in 0..self.num_layers {
                let inc_k_cache =
                    output[format!("k_cache_{}", i)].try_extract_array::<KvDType>()?;
                let inc_v_cache =
                    output[format!("v_cache_{}", i)].try_extract_array::<KvDType>()?;

                // The new data is the last row of the incremental output from the model
                let k_new_slice = inc_k_cache.slice(s![valid_len, .., ..]);
                let v_new_slice = inc_v_cache.slice(s![valid_len, .., ..]);

                // Paste the new row into our long-running cache at the correct position
                k_caches[i]
                    .slice_mut(s![valid_len, .., ..])
                    .assign(&k_new_slice);
                v_caches[i]
                    .slice_mut(s![valid_len, .., ..])
                    .assign(&v_new_slice);
            }

            // --- 4. Update valid length and check stop condition ---
            valid_len = new_valid_len;

            if idx >= 1500 || y_vec.last().map_or(false, |&v| v == T2S_DECODER_EOS) {
                let sliced = y_vec
                    .split_off(prefix_len)
                    .into_iter()
                    .map(|i| if i == T2S_DECODER_EOS { 0 } else { i })
                    .collect::<Vec<i64>>();
                let y = ArrayD::from_shape_vec(IxDyn(&[1, 1, sliced.len()]), sliced)?;
                debug!("t2s final idx: {}", idx);
                return Ok(y);
            }
            idx += 1;
        }
    }

    /// synthesize async
    ///
    /// `text` is input text for run
    ///
    /// `lang_id` can be LangId::Auto(Mandarin) or LangId::AutoYue（cantonese）
    ///
    pub async fn synthesize(
        &mut self,
        text: &str,
        sampling_param: SamplingParams,
        lang_id: LangId,
    ) -> Result<
        (
            WavSpec,
            impl Stream<Item = Result<f32, GSVError>> + Send + Unpin,
        ),
        GSVError,
    > {
        let ref_data = self
            .ref_data
            .as_ref()
            .ok_or(GSVError::from("Reference data not initialized"))?;
        let spec = self.output_spec;
        let text = ensure_punctuation(text);
        let time = SystemTime::now();
        let texts_and_seqs = self.text_processor.get_phone_and_bert(&text, lang_id)?;
        debug!("g2pw and preprocess time: {:?}", time.elapsed()?);
        let ref_data = ref_data.clone();

        let stream = stream! {
            for (text, seq, bert) in texts_and_seqs {
                match self.in_stream_once_gen(&text, &bert, &seq, &ref_data, sampling_param).await {
                    Ok(samples) => {
                        for sample in samples {
                            yield Ok(sample);
                        }
                    }
                    Err(e) => yield Err(e),
                }
            }
        };

        Ok((spec, Box::pin(stream)))
    }

    async fn in_stream_once_gen(
        &mut self,
        _text: &str,
        text_bert: &Array2<f32>,
        text_seq_vec: &[i64],
        ref_data: &ReferenceData,
        sampling_param: SamplingParams,
    ) -> Result<Vec<f32>, GSVError> {
        let text_seq = Array2::from_shape_vec((1, text_seq_vec.len()), text_seq_vec.to_vec())?;
        // let mut text_bert = Array2::<f32>::zeros((text_seq.shape()[1], 1024));
        let mut sampler = Sampler::new(VOCAB_SIZE);

        let (x, prompts) = {
            let time = SystemTime::now();
            let encoder_output = self.t2s_encoder.run(inputs![
                "ref_seq" => TensorRef::from_array_view(&ref_data.ref_seq)?,
                "text_seq" => TensorRef::from_array_view(&text_seq)?,
                "ref_bert" => TensorRef::from_array_view(&ref_data.ref_bert)?,
                "text_bert" => TensorRef::from_array_view(text_bert)?,
                "ssl_content" => TensorRef::from_array_view(&ref_data.ssl_content)?
            ])?;
            debug!("T2S Encoder time: {:?}", time.elapsed()?);
            (
                encoder_output["x"].try_extract_array::<f32>()?.into_owned(),
                encoder_output["prompts"]
                    .try_extract_array::<i64>()?
                    .into_owned(),
            )
        };
        let (mut y_vec, _) = prompts.clone().into_raw_vec_and_offset();

        let prefix_len = y_vec.len();
        let (y_vec, y_emb, x_example, k_caches, v_caches, initial_seq_len) = {
            let time = SystemTime::now();
            let fs_decoder_output = self.t2s_fs_decoder.run(inputs![
                "x" => TensorRef::from_array_view(&x)?,
                "prompts" => TensorRef::from_array_view(&prompts)?
            ])?;
            debug!("T2S FS Decoder time: {:?}", time.elapsed()?);

            let logits = fs_decoder_output["logits"]
                .try_extract_array::<f32>()?
                .into_owned();
            let y_emb = fs_decoder_output["y_emb"]
                .try_extract_array::<f32>()?
                .into_owned();
            let x_example = fs_decoder_output["x_example"]
                .try_extract_array::<f32>()?
                .into_owned();

            // --- Initialize large KV Caches ---
            // Get shape and initial data from the first-pass decoder.
            let k_init_first = fs_decoder_output["k_cache_0"].try_extract_array::<KvDType>()?;
            let initial_dims_dyn = k_init_first.raw_dim();
            let initial_seq_len = initial_dims_dyn[0];

            // Define the shape for our large, pre-allocated cache.
            let mut large_cache_dims = initial_dims_dyn.clone();
            large_cache_dims[0] = INITIAL_CACHE_SIZE;

            let mut k_caches = Vec::with_capacity(self.num_layers);
            let mut v_caches = Vec::with_capacity(self.num_layers);

            for i in 0..self.num_layers {
                let k_init = fs_decoder_output[format!("k_cache_{}", i).to_string()]
                    .try_extract_array::<KvDType>()?;
                let v_init = fs_decoder_output[format!("v_cache_{}", i).to_string()]
                    .try_extract_array::<KvDType>()?;

                // Create large, zero-initialized caches.
                let mut k_large = Array::zeros(large_cache_dims.clone());
                let mut v_large = Array::zeros(large_cache_dims.clone());

                // Copy the initial data from the first-pass decoder into the start of our large caches.
                k_large
                    .slice_mut(s![0..initial_seq_len, .., ..])
                    .assign(&k_init);
                v_large
                    .slice_mut(s![0..initial_seq_len, .., ..])
                    .assign(&v_init);

                k_caches.push(k_large);
                v_caches.push(v_large);
            }
            let (mut logits_vec, _) = logits.into_raw_vec_and_offset();
            let sampling_rst = sampler.sample(
                &mut logits_vec,
                &y_vec,
                &sampling_param,
                &[0, T2S_DECODER_EOS], // avoid breath
            );
            // debug!("sampled token {}", sampling_rst);
            y_vec.push(sampling_rst);
            (y_vec, y_emb, x_example, k_caches, v_caches, initial_seq_len)
        };

        let time = SystemTime::now();
        let pred_semantic = self.run_t2s_s_decoder_loop(
            &mut sampler,
            sampling_param,
            y_vec,
            y_emb,
            x_example,
            k_caches,
            v_caches,
            prefix_len,
            initial_seq_len,
        )?;
        debug!("T2S S Decoder all time: {:?}", time.elapsed()?);

        let time = SystemTime::now();
        let outputs = self.sovits.run(inputs![
            "text_seq" => TensorRef::from_array_view(&text_seq)?,
            "pred_semantic" => TensorRef::from_array_view(&pred_semantic)?,
            "ref_audio" => TensorRef::from_array_view(&ref_data.ref_audio_32k)?
        ])?;
        debug!("SoVITS time: {:?}", time.elapsed()?);
        let output_audio = outputs["audio"].try_extract_array::<f32>()?;
        let mut audio = output_audio.into_owned().into_raw_vec();

        for sample in &mut audio {
            *sample = *sample * 4.0;
        }

        Ok(audio)
    }

    /// synthesize
    ///
    /// `text` is input text for run
    ///
    /// `lang_id` can be LangId::Auto(Mandarin) or LangId::AutoYue（cantonese）
    ///
    pub fn synthesize_sync(
        &mut self,
        text: &str,
        sampling_param: SamplingParams,
        lang_id: LangId,
    ) -> Result<(WavSpec, Vec<f32>), GSVError> {
        Self::run_async_in_context(async {
            let (spec, stream) = self.synthesize(text, sampling_param, lang_id).await?;
            let mut samples = Vec::new();
            futures::pin_mut!(stream);
            while let Some(sample) = stream.next().await {
                samples.push(sample?);
            }
            Ok((spec, samples))
        })
    }
}

fn ensure_punctuation(text: &str) -> String {
    if !text.ends_with(['。', '.']) {
        text.to_string() + "。"
    } else {
        text.to_string()
    }
}

fn read_and_resample_audio<P: AsRef<Path>>(
    path: P,
) -> Result<(Array2<f32>, Array2<f32>), GSVError> {
    let file = File::open(&path)
        .map_err(|e| GSVError::from(format!("Failed to open reference audio: {}", e)))?;
    let wav_reader = WavReader::new(file)?;
    let spec = wav_reader.spec();
    let audio_samples: Vec<f32> = wav_reader
        .into_samples::<i16>()
        .collect::<Result<Vec<i16>, _>>()?
        .into_iter()
        .map(|s| s as f32 / i16::MAX as f32)
        .collect();

    let ref_audio_16k = if spec.sample_rate != 16000 {
        resample_audio(audio_samples.clone(), spec.sample_rate, 16000)?
    } else {
        audio_samples.clone()
    };
    let ref_audio_32k = resample_audio(audio_samples, spec.sample_rate, 32000)?;

    Ok((
        Array2::from_shape_vec((1, ref_audio_16k.len()), ref_audio_16k)?,
        Array2::from_shape_vec((1, ref_audio_32k.len()), ref_audio_32k)?,
    ))
}

fn resample_audio(input: Vec<f32>, in_rate: u32, out_rate: u32) -> Result<Vec<f32>, GSVError> {
    if in_rate == out_rate {
        return Ok(input);
    }

    let ratio = in_rate as f32 / out_rate as f32;
    let out_len = ((input.len() as f32 / ratio).ceil() as usize).max(1);
    let mut output = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let src_idx = i as f32 * ratio;
        let idx_floor = src_idx.floor() as usize;
        let frac = src_idx - idx_floor as f32;

        output.push(if idx_floor + 1 < input.len() {
            input[idx_floor] * (1.0 - frac) + input[idx_floor + 1] * frac
        } else if idx_floor < input.len() {
            input[idx_floor]
        } else {
            0.0
        });
    }

    Ok(output)
}
