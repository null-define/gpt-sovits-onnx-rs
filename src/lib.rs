use async_stream::stream;
use futures::{Stream, StreamExt};
use hound::{WavReader, WavSpec};
use jieba_rs::Jieba;
use log::{debug, error, info};
use ndarray::{
    Array1, Array2, ArrayBase, Axis, Dim, IxDyn, IxDynImpl, OwnedRepr, Slice, concatenate, s,
};
use ort::{
    execution_providers::{CPUExecutionProvider, XNNPACKExecutionProvider},
    inputs,
    session::{Session, builder::GraphOptimizationLevel},
    value::{Tensor, TensorRef, Value},
};
use std::sync::Arc;
use std::time::SystemTime;
use std::{fs::File, path::Path, str::FromStr};
use tokenizers::Tokenizer;
use tokio::task::block_in_place;

mod error;
mod text;
mod utils;

use crate::text::{TextProcessor, g2pw::G2PWConverter};
pub use error::GSVError;

const EARLY_STOP_NUM: usize = 1500;
const T2S_DECODER_EOS: i64 = 1024;
type KvDType = f32;

static BERT_TOKENIZER: &str = include_str!("../resource/g2pw_tokenizer.json");

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
    bert: Option<Session>,
    bert_tokenizer: Arc<Tokenizer>,
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
        bert_path: Option<P>,
    ) -> Result<Self, GSVError> {
        info!("Initializing TTSModel with ONNX sessions");

        let create_session = |path: P| {
            Session::builder()?
                .with_execution_providers([CPUExecutionProvider::default()
                    .with_arena_allocator(true)
                    .build()])?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(8)?
                .commit_from_file(path)
        };

        Ok(TTSModel {
            text_processor: TextProcessor {
                jieba: Jieba::new(),
                g2pw: G2PWConverter::new(
                    create_session(g2pw_path)?,
                    Arc::new(Tokenizer::from_str(BERT_TOKENIZER).unwrap()),
                )?,
                symbols: text::symbols::SYMBOLS.clone(),
            },
            sovits: create_session(sovits_path)?,
            ssl: create_session(ssl_path)?,
            t2s_encoder: create_session(t2s_encoder_path)?,
            t2s_fs_decoder: create_session(t2s_fs_decoder_path)?,
            t2s_s_decoder: create_session(t2s_s_decoder_path)?,
            bert: bert_path.map(|p| create_session(p)).transpose()?,
            bert_tokenizer: Arc::new(Tokenizer::from_str(BERT_TOKENIZER).unwrap()),
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
        let ref_text = ensure_punctuation(ref_text);
        let phones = self.text_processor.get_phone(&ref_text)?;
        let (ref_seq, word2ph): (Vec<i64>, Vec<i32>) =
            phones
                .into_iter()
                .fold((Vec::new(), Vec::new()), |(mut seq, mut w2ph), p| {
                    seq.extend(p.2);
                    w2ph.extend(p.1);
                    (seq, w2ph)
                });

        let ref_seq = Array2::from_shape_vec((1, ref_seq.len()), ref_seq)?;
        let mut ref_bert = Array2::<f32>::zeros((ref_seq.shape()[1], 1024));

        if let Some(_) = self.bert.as_mut() {
            ref_bert = self.process_bert(&ref_text, &word2ph)?;
        }

        let (ref_audio_16k, ref_audio_32k) = read_and_resample_audio(&reference_audio_path)?;
        let ssl_content = self.process_ssl(&ref_audio_16k)?;

        self.ref_data = Some(ReferenceData {
            ref_seq,
            ref_bert,
            ref_audio_32k,
            ssl_content,
        });

        Ok(())
    }

    fn process_bert(&mut self, text: &str, word2ph: &[i32]) -> Result<Array2<f32>, GSVError> {
        let encoding = self.bert_tokenizer.encode(text, true).unwrap();
        let (input_ids, attention_mask, token_type_ids): (Vec<i64>, Vec<i64>, Vec<i64>) = (
            encoding.get_ids().iter().map(|&id| id as i64).collect(),
            encoding
                .get_attention_mask()
                .iter()
                .map(|&m| m as i64)
                .collect(),
            encoding.get_type_ids().iter().map(|&t| t as i64).collect(),
        );

        let inputs = inputs![
            "input_ids" => Tensor::from_array(Array2::from_shape_vec((1, input_ids.len()), input_ids).unwrap()).unwrap(),
            "attention_mask" => Tensor::from_array(Array2::from_shape_vec((1, attention_mask.len()), attention_mask).unwrap()).unwrap(),
            "token_type_ids" => Tensor::from_array(Array2::from_shape_vec((1, token_type_ids.len()), token_type_ids).unwrap()).unwrap()
        ];

        let bert_out = self.bert.as_mut().unwrap().run(inputs)?;
        let bert_feature = bert_out["bert_feature"]
            .try_extract_array::<f32>()?
            .into_owned();
        let bert_feature_2d: Array2<f32> = bert_feature.into_dimensionality()?;
        Ok(build_phone_level_feature(
            bert_feature_2d,
            Array1::from_vec(word2ph.to_vec()),
        ))
    }

    fn process_ssl(
        &mut self,
        ref_audio_16k: &Array2<f32>,
    ) -> Result<ArrayBase<OwnedRepr<f32>, IxDyn>, GSVError> {
        let time = SystemTime::now();
        let ssl_output = self
            .ssl
            .run(inputs!["ref_audio_16k" => TensorRef::from_array_view(ref_audio_16k).unwrap()])?;
        info!("SSL processing time: {:?}", time.elapsed()?);
        Ok(ssl_output["ssl_content"]
            .try_extract_array::<f32>()?
            .into_owned())
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
        mut y: ArrayBase<OwnedRepr<i64>, IxDyn>,
        mut y_emb: ArrayBase<OwnedRepr<f32>, IxDyn>,
        x_example: ArrayBase<OwnedRepr<f32>, IxDyn>,
        mut k_caches: Vec<ArrayBase<OwnedRepr<KvDType>, IxDyn>>,
        mut v_caches: Vec<ArrayBase<OwnedRepr<KvDType>, IxDyn>>,
        prefix_len: usize,
    ) -> Result<ArrayBase<OwnedRepr<i64>, IxDyn>, GSVError> {
        let mut idx = 1;
        loop {
            let time = SystemTime::now();
            let mut inputs = inputs![
                "iy" => TensorRef::from_array_view(&y).unwrap(),
                "iy_emb" => TensorRef::from_array_view(&y_emb).unwrap(),
                "x_example" => TensorRef::from_array_view(&x_example).unwrap()
            ];

            for i in 0..self.num_layers {
                inputs.push((
                    format!("ik_cache_{}", i).into(),
                    TensorRef::from_array_view(&k_caches[i])?.into(),
                ));
                inputs.push((
                    format!("iv_cache_{}", i).into(),
                    TensorRef::from_array_view(&v_caches[i])?.into(),
                ));
            }

            let output = self.t2s_s_decoder.run(inputs)?;
            debug!(
                "T2S S Decoder iteration {} time: {:?}",
                idx,
                time.elapsed()?
            );

            y = output["y"].try_extract_array::<i64>()?.into_owned();
            y_emb = output["y_emb"].try_extract_array::<f32>()?.into_owned();
            for i in 0..self.num_layers {
                k_caches[i] = output[format!("k_cache_{}", i)]
                    .try_extract_array::<KvDType>()?
                    .into_owned();
                v_caches[i] = output[format!("v_cache_{}", i)]
                    .try_extract_array::<KvDType>()?
                    .into_owned();
            }

            if idx > 10
                && (idx >= 1500
                    || (y.shape()[1] - prefix_len) > EARLY_STOP_NUM
                    || y.last().map_or(false, |&v| v == T2S_DECODER_EOS))
            {
                let seq_len = y.shape()[1];
                return Ok(y
                    .slice_axis(Axis(1), Slice::from(prefix_len + 5..seq_len))
                    .map(|&i| if i == T2S_DECODER_EOS { 0 } else { i })
                    .insert_axis(Axis(0)));
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
        let ref_data = self
            .ref_data
            .as_ref()
            .ok_or(GSVError::from("Reference data not initialized"))?;
        let spec = WavSpec {
            channels: 1,
            sample_rate: 32000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let text = ensure_punctuation(text);
        let texts_and_seqs = self.text_processor.get_phone(&text)?;
        let ref_data = ref_data.clone();

        let stream = stream! {
            for (text, word2ph, seq) in texts_and_seqs {
                match self.in_stream_once_gen(&text, &word2ph, &seq, &ref_data).await {
                    Ok(samples) => {
                        debug!("Yielding {} samples", samples.len());
                        for sample in samples {
                            yield Ok(sample * 4.0);
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
        text: &str,
        word2ph: &[i32],
        text_seq_vec: &[i64],
        ref_data: &ReferenceData,
    ) -> Result<Vec<f32>, GSVError> {
        let text_seq = Array2::from_shape_vec((1, text_seq_vec.len()), text_seq_vec.to_vec())?;
        let mut text_bert = Array2::<f32>::zeros((text_seq.shape()[1], 1024));
        if let Some(_) = self.bert.as_mut() {
            text_bert = self.process_bert(&text, &word2ph)?;
        }

        let (x, prompts) = {
            let time = SystemTime::now();
            let encoder_output = self.t2s_encoder.run(inputs![
                "ref_seq" => TensorRef::from_array_view(&ref_data.ref_seq)?,
                "text_seq" => TensorRef::from_array_view(&text_seq)?,
                "ref_bert" => TensorRef::from_array_view(&ref_data.ref_bert)?,
                "text_bert" => TensorRef::from_array_view(&text_bert)?,
                "ssl_content" => TensorRef::from_array_view(&ref_data.ssl_content)?
            ])?;
            info!("T2S Encoder time: {:?}", time.elapsed()?);
            (
                encoder_output["x"].try_extract_array::<f32>()?.into_owned(),
                encoder_output["prompts"]
                    .try_extract_array::<i64>()?
                    .into_owned(),
            )
        };

        let prefix_len = prompts.dim()[1];
        let (y, y_emb, x_example, k_caches, v_caches) = {
            let time = SystemTime::now();
            let fs_decoder_output = self.t2s_fs_decoder.run(inputs![
                "x" => TensorRef::from_array_view(&x)?,
                "prompts" => TensorRef::from_array_view(&prompts)?
            ])?;
            info!("T2S FS Decoder time: {:?}", time.elapsed()?);

            let y = fs_decoder_output["y"]
                .try_extract_array::<i64>()?
                .into_owned();
            let y_emb = fs_decoder_output["y_emb"]
                .try_extract_array::<f32>()?
                .into_owned();
            let x_example = fs_decoder_output["x_example"]
                .try_extract_array::<f32>()?
                .into_owned();
            let k_caches = (0..self.num_layers)
                .map(|i| {
                    fs_decoder_output[format!("k_cache_{}", i)]
                        .try_extract_array::<KvDType>()
                        .unwrap()
                        .into_owned()
                })
                .collect::<Vec<ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>>>();
            let v_caches = (0..self.num_layers)
                .map(|i| {
                    fs_decoder_output[format!("v_cache_{}", i)]
                        .try_extract_array::<KvDType>()
                        .unwrap()
                        .into_owned()
                })
                .collect::<Vec<ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>>>();
            (y, y_emb, x_example, k_caches, v_caches)
        };

        let time = SystemTime::now();
        let pred_semantic =
            self.run_t2s_s_decoder_loop(y, y_emb, x_example, k_caches, v_caches, prefix_len)?;
        info!("T2S S Decoder loop time: {:?}", time.elapsed()?);

        let time = SystemTime::now();
        let outputs = self.sovits.run(inputs![
            "text_seq" => TensorRef::from_array_view(&text_seq)?,
            "pred_semantic" => TensorRef::from_array_view(&pred_semantic)?,
            "ref_audio" => TensorRef::from_array_view(&ref_data.ref_audio_32k)?
        ])?;
        info!("SoVITS time: {:?}", time.elapsed()?);
        Ok(outputs["audio"]
            .try_extract_array::<f32>()?
            .into_owned()
            .into_raw_vec())
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

fn ensure_punctuation(text: &str) -> String {
    if !text.ends_with(['。', '.']) {
        text.to_string() + "."
    } else {
        text.to_string()
    }
}

fn build_phone_level_feature(res: Array2<f32>, word2ph: Array1<i32>) -> Array2<f32> {
    let phone_level_features = word2ph
        .into_iter()
        .enumerate()
        .map(|(i, count)| {
            let row = res.row(i);
            Array2::from_shape_fn((count as usize, res.ncols()), |(j, k)| row[k])
        })
        .collect::<Vec<_>>();
    concatenate(
        Axis(0),
        &phone_level_features
            .iter()
            .map(|x| x.view())
            .collect::<Vec<_>>(),
    )
    .unwrap()
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
