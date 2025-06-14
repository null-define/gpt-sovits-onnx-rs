use async_stream::stream;
use futures::{StreamExt, stream::Stream};
use hound::{WavReader, WavSpec};
use jieba_rs::Jieba;
use log::{debug, error, info};
use ndarray::{Array, ArrayBase, Axis, Dim, IxDyn, IxDynImpl, OwnedRepr, Slice, ViewRepr, s};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::{TensorRef, Value};
use ort::{
    execution_providers::{CPUExecutionProvider, XNNPACKExecutionProvider},
    inputs,
};
use std::num::NonZero;
use std::str::FromStr;
use std::sync::Arc;
use std::{fs::File, path::Path, time::SystemTime};
use tokio::task::block_in_place;

mod error;
mod text;
mod utils;

use crate::text::TextProcessor;
use crate::text::g2pw::G2PWConverter;
pub use error::GSVError;

const EARLY_STOP_NUM: usize = 1500; // Match old code's max iteration limit
const T2S_DECODER_EOS: usize = 1024; // Assuming EOS token remains consistent

type InternalDType = f32; // Verify with model specs; f16 may be needed if model expects it

// Struct to hold reference data
#[derive(Clone)]
pub struct ReferenceData {
    ref_seq: Array<i64, Dim<[usize; 2]>>,
    ref_bert: Array<f32, Dim<[usize; 2]>>,
    ref_audio_32k: Array<f32, Dim<[usize; 2]>>,
    ssl_content: Array<f32, IxDyn>,
}

// Struct to hold model sessions and reference data
pub struct TTSModel {
    text_processor: TextProcessor,
    sovits: Session,
    ssl: Session,
    t2s_encoder: Session,
    t2s_fs_decoder: Session,
    t2s_s_decoder: Session,
    ref_data: Option<ReferenceData>,
    num_layers: usize,
}

impl TTSModel {
    pub fn new<P: AsRef<Path>>(
        g2pw_path: P,
        sovits_path: P,
        ssl_path: P,
        t2s_encoder_path: P,
        t2s_fs_decoder_path: P,
        t2s_s_decoder_path: P,
        num_layers: usize,
    ) -> Result<Self, GSVError> {
        info!("Initializing TTSModel with ONNX sessions");
        // let cpu_session_config = || {
        //     Session::builder()?
        //         .with_execution_providers([CPUExecutionProvider::default()
        //             .with_arena_allocator(true)
        //             .build()])?
        //         .with_optimization_level(GraphOptimizationLevel::Level3)?
        //         .with_intra_threads(8)
        // };

        let cpu_session2_config = || {
            Session::builder()?
                .with_execution_providers([CPUExecutionProvider::default()
                    .with_arena_allocator(true)
                    .build()])?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(4)?
                .with_inter_threads(2)
        };

        let xnnpack_session_config = || {
            Session::builder()?
                .with_execution_providers([XNNPACKExecutionProvider::default()
                    .with_intra_op_num_threads(NonZero::new(8).unwrap())
                    .build()])?
                .with_optimization_level(GraphOptimizationLevel::Level3)
        };
        let sovits = xnnpack_session_config()?.commit_from_file(sovits_path)?;
        let ssl = xnnpack_session_config()?.commit_from_file(ssl_path)?;
        let t2s_encoder = cpu_session2_config()?.commit_from_file(t2s_encoder_path)?;
        let t2s_fs_decoder = cpu_session2_config()?.commit_from_file(t2s_fs_decoder_path)?;
        let t2s_s_decoder = cpu_session2_config()?
            // .with_profiling("d2s")?
            .commit_from_file(t2s_s_decoder_path)?;

        let g2pw_session = xnnpack_session_config()?.commit_from_file(g2pw_path)?;

        let tokenizer =
            Arc::new(tokenizers::Tokenizer::from_str(text::g2pw::G2PW_TOKENIZER).unwrap());

        let text_processor = TextProcessor {
            jieba: Jieba::new(),
            g2pw: G2PWConverter::new(g2pw_session, tokenizer).unwrap(),
            symbols: text::symbols::SYMBOLS.clone(),
        };

        Ok(TTSModel {
            text_processor,
            sovits,
            ssl,
            t2s_encoder,
            t2s_fs_decoder,
            t2s_s_decoder,
            ref_data: None,
            num_layers,
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

    pub async fn process_reference<P: AsRef<Path>>(
        &mut self,
        reference_audio_path: P,
        ref_text: &str,
    ) -> Result<(), GSVError> {
        info!("Processing reference audio and text: {}", ref_text);
        let ref_text = if !ref_text.ends_with(['。', '.']) {
            ref_text.to_string() + "."
        } else {
            ref_text.to_string()
        };
        let ref_seq = self.text_processor.get_phone(&ref_text)?.concat();
        debug!("Reference phoneme: {:?}", ref_seq);
        let ref_seq = Array::from_shape_vec((1, ref_seq.len()), ref_seq)
            .map_err(|e| GSVError::from(format!("Failed to create ref_seq array: {}", e)))?;
        let ref_bert = Array::<f32, _>::zeros((ref_seq.shape()[1], 1024));

        let file = File::open(&reference_audio_path)
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
        let ref_audio_16k = Array::from_shape_vec((1, ref_audio_16k.len()), ref_audio_16k)
            .map_err(|e| GSVError::from(format!("Failed to create ref_audio_16k array: {}", e)))?;

        let ref_audio_32k = resample_audio(audio_samples, spec.sample_rate, 32000)?;
        let ref_audio_32k = Array::from_shape_vec((1, ref_audio_32k.len()), ref_audio_32k)
            .map_err(|e| GSVError::from(format!("Failed to create ref_audio_32k array: {}", e)))?;

        let time = SystemTime::now();
        let ssl_output = self
            .ssl
            .run(ort::inputs!["ref_audio_16k" => TensorRef::from_array_view(&ref_audio_16k)?])?;
        info!("SSL processing time: {:?}", time.elapsed()?);
        let ssl_content = ssl_output["ssl_content"]
            .try_extract_array::<f32>()?
            .into_owned();

        self.ref_data = Some(ReferenceData {
            ref_seq,
            ref_bert,
            ref_audio_32k,
            ssl_content,
        });

        Ok(())
    }

    pub fn process_reference_sync<P: AsRef<Path>>(
        &mut self,
        reference_audio_path: P,
        ref_text: &str,
    ) -> Result<(), GSVError> {
        Self::run_async_in_context(self.process_reference(reference_audio_path, ref_text))
    }

    fn run_t2s_s_decoder_loop(
        &mut self,
        mut y: Array<i64, Dim<IxDynImpl>>,
        mut y_emb: Array<InternalDType, IxDyn>,
        x_example: Array<InternalDType, IxDyn>,
        mut k_caches_arr: Vec<Array<InternalDType, IxDyn>>,
        mut v_caches_arr: Vec<Array<InternalDType, IxDyn>>,
        prefix_len: usize,
    ) -> Result<ArrayBase<OwnedRepr<i64>, Dim<IxDynImpl>>, GSVError> {
        let x_example_val = Value::from_array(x_example)?;
        let mut idx = 1;
        loop {
            let time = SystemTime::now();
            let mut s_decoder_inputs = ort::inputs![
                "iy" => TensorRef::from_array_view(&y)?,
                "iy_emb" => TensorRef::from_array_view(&y_emb)?,
                "x_example" => x_example_val.clone(),
            ];
            for i in 0..self.num_layers {
                s_decoder_inputs.push((
                    format!("ik_cache_{}", i).into(),
                    TensorRef::from_array_view(&k_caches_arr[i])?.into(),
                ));
                s_decoder_inputs.push((
                    format!("iv_cache_{}", i).into(),
                    TensorRef::from_array_view(&v_caches_arr[i])?.into(),
                ));
            }
            let s_decoder_output = self.t2s_s_decoder.run(s_decoder_inputs)?;
            debug!(
                "T2S S Decoder iteration {} time: {:?}",
                idx,
                time.elapsed()?
            );

            y = s_decoder_output["y"]
                .try_extract_array::<i64>()?
                .into_owned();
            y_emb = s_decoder_output["y_emb"]
                .try_extract_array::<InternalDType>()?
                .into_owned();

            for i in 0..self.num_layers {
                k_caches_arr[i] = s_decoder_output[format!("k_cache_{}", i)]
                    .try_extract_array::<InternalDType>()?
                    .into_owned();
                v_caches_arr[i] = s_decoder_output[format!("v_cache_{}", i)]
                    .try_extract_array::<InternalDType>()?
                    .into_owned();
            }

            debug!("S Decoder iteration {}: y shape = {:?}", idx, y.shape());

            if idx > 10
                && (idx >= 1500
                    || (y.shape()[1] - prefix_len) > EARLY_STOP_NUM
                    || y.last().unwrap_or(&(T2S_DECODER_EOS as i64)) == &(T2S_DECODER_EOS as i64))
            {
                let seq_len = y.shape()[1];
                let pred_semantic = y
                    .slice_axis(Axis(1), Slice::from(prefix_len + 1..seq_len))
                    .into_owned()
                    .map(|i| {
                        if *i == (T2S_DECODER_EOS as i64) {
                            0
                        } else {
                            *i
                        }
                    });
                return Ok(pred_semantic.insert_axis(Axis(0)));
            }

            idx += 1;
        }
    }

    pub async fn run(
        &mut self,
        text: &str,
    ) -> Result<
        (
            WavSpec,
            impl Stream<Item = Result<f32, GSVError>> + Send + Unpin,
        ),
        GSVError,
    > {
        info!("Running inference for text: {}", text);
        let ref_data = self
            .ref_data
            .as_ref()
            .ok_or_else(|| GSVError::from("Reference data not initialized"))?;

        let spec = WavSpec {
            channels: 1,
            sample_rate: 32000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        let text = if !text.ends_with(['。', '.']) {
            text.to_string() + "."
        } else {
            text.to_string()
        };
        let text_seqs = self.text_processor.get_phone(&text)?;

        let ref_data = ref_data.clone();

        let stream = stream! {
            for text_seq_vec in text_seqs {
                match self.in_stream_once_gen(&text_seq_vec, &ref_data).await {
                    Ok(samples) => {
                        debug!("Yielding {} samples for normalized text", samples.len());
                        for sample in samples {
                            yield Ok(sample * 4.0);
                        }
                    }
                    Err(e) => {
                        error!("Error in in_stream_once_gen: {}", e);
                        yield Err(e);
                    }
                }
            }
        };

        Ok((spec, Box::pin(stream)))
    }

    async fn in_stream_once_gen(
        &mut self,
        text_seq_vec: &Vec<i64>,
        ref_data: &ReferenceData,
    ) -> Result<Vec<f32>, GSVError> {
        debug!("Text phoneme: {:?}", text_seq_vec);
        let num_layers = self.num_layers;

        let text_seq = Array::from_shape_vec((1, text_seq_vec.len()), text_seq_vec.to_vec())
            .map_err(|e| GSVError::from(format!("Failed to create text_seq array: {}", e)))?;
        let text_bert = Array::<f32, _>::zeros((text_seq.shape()[1], 1024));

        let time = SystemTime::now();
        let x_owned: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>;
        let prompts_owned: ArrayBase<OwnedRepr<i64>, Dim<IxDynImpl>>;
        {
            let encoder_output = self.t2s_encoder.run(ort::inputs![
                "ref_seq" => TensorRef::from_array_view(&ref_data.ref_seq)?,
                "text_seq" => TensorRef::from_array_view(&text_seq)?,
                "ref_bert" => TensorRef::from_array_view(&ref_data.ref_bert)?,
                "text_bert" => TensorRef::from_array_view(&text_bert)?,
                "ssl_content" => TensorRef::from_array_view(&ref_data.ssl_content)?
            ])?;
            info!("T2S Encoder time: {:?}", time.elapsed()?);
            x_owned = encoder_output["x"].try_extract_array::<f32>()?.into_owned();
            prompts_owned = encoder_output["prompts"]
                .try_extract_array::<i64>()?
                .into_owned();
        }

        let prefix_len = prompts_owned.dim()[1];

        let time = SystemTime::now();
        let (y, y_emb, x_example, initial_k_caches_arr, initial_v_caches_arr) = {
            let fs_decoder_inputs = ort::inputs![
                "x" => TensorRef::from_array_view(&x_owned)?,
                "prompts" => TensorRef::from_array_view(&prompts_owned)?
            ];
            let fs_decoder_output = self.t2s_fs_decoder.run(fs_decoder_inputs)?;
            info!("T2S FS Decoder time: {:?}", time.elapsed()?);

            let y = fs_decoder_output["y"]
                .try_extract_array::<i64>()?
                .into_owned();
            let y_emb = fs_decoder_output["y_emb"]
                .try_extract_array::<InternalDType>()?
                .into_owned();
            let x_example = fs_decoder_output["x_example"]
                .try_extract_array::<InternalDType>()?
                .into_owned();

            let initial_k_caches_arr: Vec<Array<InternalDType, IxDyn>> = (0..num_layers)
                .map(|i| {
                    fs_decoder_output[format!("k_cache_{}", i)]
                        .try_extract_array::<InternalDType>()
                        .map(|arr| arr.into_owned())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let initial_v_caches_arr: Vec<Array<InternalDType, IxDyn>> = (0..num_layers)
                .map(|i| {
                    fs_decoder_output[format!("v_cache_{}", i)]
                        .try_extract_array::<InternalDType>()
                        .map(|arr| arr.into_owned())
                })
                .collect::<Result<Vec<_>, _>>()?;
            (
                y,
                y_emb,
                x_example,
                initial_k_caches_arr,
                initial_v_caches_arr,
            )
        };
        let time = SystemTime::now();
        let pred_semantic = self.run_t2s_s_decoder_loop(
            y,
            y_emb,
            x_example,
            initial_k_caches_arr,
            initial_v_caches_arr,
            prefix_len,
        )?;

        info!(
            "T2S S Decoder loop finished. time cost {:?}",
            time.elapsed()?
        );

        let time = SystemTime::now();
        let outputs = self.sovits.run(ort::inputs![
            "text_seq" => TensorRef::from_array_view(&text_seq)?,
            "pred_semantic" => TensorRef::from_array_view(&pred_semantic)?,
            "ref_audio" => TensorRef::from_array_view(&ref_data.ref_audio_32k)?
        ])?;
        let output = outputs["audio"].try_extract_array::<f32>()?;
        info!("SoVITS time: {:?}", time.elapsed()?);
        let samples = output.into_owned().into_raw_vec();
        Ok(samples)
    }

    pub fn run_sync(&mut self, text: &str) -> Result<(WavSpec, Vec<f32>), GSVError> {
        Self::run_async_in_context(async {
            let (spec, stream) = self.run(text).await?;
            let mut samples = Vec::new();
            futures::pin_mut!(stream);
            while let Some(sample) = stream.next().await {
                samples.push(sample?);
            }
            Ok((spec, samples))
        })
    }
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

        if idx_floor + 1 < input.len() {
            let sample = input[idx_floor] * (1.0 - frac) + input[idx_floor + 1] * frac;
            output.push(sample);
        } else if idx_floor < input.len() {
            output.push(input[idx_floor]);
        } else {
            output.push(0.0);
        }
    }

    Ok(output)
}
