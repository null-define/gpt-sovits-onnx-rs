use async_stream::stream;
use futures::{StreamExt, stream::Stream};
use hound::{WavReader, WavSpec};
use jieba_rs::Jieba;
use log::{debug, error, info};
use ndarray::{Array, Axis, Dim, IntoDimension, IxDyn, s};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::{execution_providers::CPUExecutionProvider, value::Tensor};
use std::str::FromStr;
use std::sync::Arc;
use std::{fs::File, path::Path, time::SystemTime};
use tokio::task::block_in_place;

mod error;
mod text;
mod utils;

use utils::*;

use crate::text::TextProcessor;
use crate::text::g2pw::G2PWConverter;
pub use error::GSVError;

const EARLY_STOP_NUM: usize = 1500; // Updated to match Python's max iteration limit
const T2S_DECODER_EOS: usize = 1024; // Assuming EOS token remains consistent

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
    num_layers: usize, // Added to track number of layers for k/v caches
}

impl TTSModel {
    // Initialize the model with ONNX sessions
    pub fn new<P: AsRef<Path>>(
        g2pw_path: P,
        sovits_path: P,
        ssl_path: P,
        t2s_encoder_path: P,
        t2s_fs_decoder_path: P,
        t2s_s_decoder_path: P,
        num_layers: usize, // Added to match Python's num_layers
    ) -> Result<Self, GSVError> {
        info!("Initializing TTSModel with ONNX sessions");
        // let g2pw = Session::builder()?
        //     .with_execution_providers([CPUExecutionProvider::default().build()])?
        //     .with_optimization_level(GraphOptimizationLevel::Level3)?
        //     .with_memory_pattern(false)?
        //     .commit_from_file(g2pw_path)?;
        let sovits = Session::builder()?
            .with_execution_providers([CPUExecutionProvider::default().build()])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_memory_pattern(false)?
            .commit_from_file(sovits_path)?;
        let ssl = Session::builder()?
            .with_execution_providers([CPUExecutionProvider::default().build()])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_memory_pattern(false)?
            .commit_from_file(ssl_path)?;
        let t2s_encoder = Session::builder()?
            .with_execution_providers([CPUExecutionProvider::default().build()])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_memory_pattern(false)?
            .commit_from_file(t2s_encoder_path)?;
        let t2s_fs_decoder = Session::builder()?
            .with_execution_providers([CPUExecutionProvider::default().build()])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_memory_pattern(false)?
            .commit_from_file(t2s_fs_decoder_path)?;
        let t2s_s_decoder = Session::builder()?
            .with_execution_providers([CPUExecutionProvider::default().build()])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_inter_threads(2)?
            .with_intra_threads(4)?
            .with_parallel_execution(true)?
            .with_memory_pattern(false)?
            .commit_from_file(t2s_s_decoder_path)?;
        let tokenizer =
            Arc::new(tokenizers::Tokenizer::from_str(text::g2pw::G2PW_TOKENIZER).unwrap());

        let text_processor = TextProcessor {
            jieba: Jieba::new(),
            g2pw: G2PWConverter::new(g2pw_path, tokenizer)?,
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

    // Common function to run async code synchronously or asynchronously
    fn run_async_in_context<F, T>(fut: F) -> Result<T, GSVError>
    where
        F: std::future::Future<Output = Result<T, GSVError>>,
    {
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => block_in_place(|| handle.block_on(fut)),
            Err(_) => {
                let rt = tokio::runtime::Runtime::new()?;
                rt.block_on(fut)
            }
        }
    }

    // Process reference audio and text (async)
    pub async fn process_reference<P: AsRef<Path>>(
        &mut self,
        reference_audio_path: P,
        ref_text: &str,
    ) -> Result<(), GSVError> {
        info!("Processing reference audio and text: {}", ref_text);
        // Text processing
        // let mut extractor = PhonemeExtractor::default();
        // extractor.push_str(ref_text);
        // let ref_seq = extractor.get_phone_ids();
        let ref_seq = self.text_processor.get_phone(ref_text)?.concat();
        debug!("Reference phoneme: {:?}", ref_seq);
        let ref_seq = Array::from_shape_vec((1, ref_seq.len()), ref_seq)
            .map_err(|e| GSVError::from(format!("Failed to create ref_seq array: {}", e)))?;
        let ref_bert = Array::<f32, _>::zeros((ref_seq.shape()[1], 1024));

        // Read and resample reference audio
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

        // SSL processing
        let time = SystemTime::now();
        let ssl_output = self
            .ssl
            .run(ort::inputs!["ref_audio_16k" => Tensor::from_array(ref_audio_16k.clone())?])?;
        info!("SSL processing time: {:?}", time.elapsed()?);
        let ssl_content = ssl_output["ssl_content"]
            .try_extract_array::<f32>()?
            .to_owned();

        // Store reference data
        self.ref_data = Some(ReferenceData {
            ref_seq,
            ref_bert,
            ref_audio_32k,
            ssl_content,
        });

        Ok(())
    }

    // Synchronous wrapper for process_reference
    pub fn process_reference_sync<P: AsRef<Path>>(
        &mut self,
        reference_audio_path: P,
        ref_text: &str,
    ) -> Result<(), GSVError> {
        Self::run_async_in_context(self.process_reference(reference_audio_path, ref_text))
    }

    // Run inference with given text and return a stream (async)
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

        // Define WAV specification
        let spec = WavSpec {
            channels: 1,
            sample_rate: 32000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        // Create a stream for all normalized texts
        let text_seqs = self.text_processor.get_phone(text.into())?;

        let ref_data = ref_data.clone();
        let num_layers = self.num_layers;
        let stream = stream! {
            for text_seq in text_seqs {
                // Text processing
                debug!("Text phoneme: {:?}", text_seq);
                // let text_seq = extractor.get_phone_ids();
                // debug!("Text phoneme extractor: {:?}", extractor);
                let text_seq = match Array::from_shape_vec((1, text_seq.len()), text_seq) {
                    Ok(arr) => arr,
                    Err(e) => {
                        error!("Failed to create text_seq array': {}", e);
                        yield Err(GSVError::from(format!("Failed to create text_seq array: {}", e)));
                        return;
                    }
                };
                let text_bert = Array::<f32, _>::zeros((text_seq.shape()[1], 1024));

                // T2S Encoder
                let time = SystemTime::now();
                let encoder_output = match self.t2s_encoder.run(ort::inputs![
                    "ref_seq" => Tensor::from_array(ref_data.ref_seq.to_owned()).unwrap(),
                    "text_seq" => Tensor::from_array(text_seq.to_owned()).unwrap(),
                    "ref_bert" => Tensor::from_array(ref_data.ref_bert.to_owned()).unwrap(),
                    "text_bert" => Tensor::from_array(text_bert.to_owned()).unwrap(),
                    "ssl_content" => Tensor::from_array(ref_data.ssl_content.to_owned()).unwrap()
                ]) {
                    Ok(output) => output,
                    Err(e) => {
                        error!("T2S Encoder failed: {}", e);
                        yield Err(GSVError::from(format!("T2S Encoder failed: {}", e)));
                        return;
                    }
                };
                info!("T2S Encoder time: {:?}", time.elapsed().unwrap());
                let x = encoder_output["x"].try_extract_array::<f32>().unwrap().to_owned();
                let prompts = encoder_output["prompts"]
                    .try_extract_array::<i64>().unwrap()
                    .to_owned();
                let prefix_len = prompts.dim()[1];

                // T2S FS Decoder
                let time = SystemTime::now();
                let fs_decoder_inputs = ort::inputs![
                    "x" => Tensor::from_array(x).unwrap(),
                    "prompts" => Tensor::from_array(prompts).unwrap()
                ];
                // Initialize k_cache and v_cache for each layer
                // for i in 0..num_layers {
                //     let k_cache = Array::<f32, _>::zeros((0, 1, 512));
                //     let v_cache = Array::<f32, _>::zeros((0, 1, 512));
                //     fs_decoder_inputs.push((format!("k_cache_{}", i).into(), Tensor::from_array(k_cache).unwrap().into()));
                //     fs_decoder_inputs.push((format!("v_cache_{}", i).into(), Tensor::from_array(v_cache).unwrap().into()));
                // }
                let fs_decoder_output = match self.t2s_fs_decoder.run(fs_decoder_inputs) {
                    Ok(output) => output,
                    Err(e) => {
                        error!("T2S FS Decoder failed: {}", e);
                        yield Err(GSVError::from(format!("T2S FS Decoder failed: {}", e)));
                        return;
                    }
                };
                info!("T2S FS Decoder time: {:?}", time.elapsed().unwrap());
                let mut y = fs_decoder_output["y"].try_extract_array::<i64>().unwrap().to_owned();
                let mut k_caches: Vec<Array<f32, IxDyn>> = (0..num_layers)
                    .map(|i| fs_decoder_output[format!("k_cache_{}", i)].try_extract_array::<f32>().unwrap().to_owned())
                    .collect();
                let mut v_caches: Vec<Array<f32, IxDyn>> = (0..num_layers)
                    .map(|i| fs_decoder_output[format!("v_cache_{}", i)].try_extract_array::<f32>().unwrap().to_owned())
                    .collect();
                let mut y_emb = fs_decoder_output["y_emb"].try_extract_array::<f32>().unwrap().to_owned();
                let x_example = fs_decoder_output["x_example"].try_extract_array::<f32>().unwrap().to_owned();

                // T2S S Decoder
                let time = SystemTime::now();
                let mut idx = 1;
                let pred_semantic = loop {
                    let mut s_decoder_inputs = ort::inputs![
                        "y.1" => Tensor::from_array(y.to_owned()).unwrap(),
                        "y_emb.1" => Tensor::from_array(y_emb.to_owned()).unwrap(),
                        "x_example" => Tensor::from_array(x_example.to_owned()).unwrap()
                    ];
                    for i in 0..num_layers {
                        s_decoder_inputs.push((format!("k_cache_{}.1", i).into(), Tensor::from_array(k_caches[i].to_owned()).unwrap().into()));
                        s_decoder_inputs.push((format!("v_cache_{}.1", i).into(), Tensor::from_array(v_caches[i].to_owned()).unwrap().into()));
                    }
                    let s_decoder_output = match self.t2s_s_decoder.run(s_decoder_inputs) {
                        Ok(output) => output,
                        Err(e) => {
                            error!("T2S S Decoder failed: {}", e);
                            yield Err(GSVError::from(format!("T2S S Decoder failed: {}", e)));
                            return;
                        }
                    };

                    y = s_decoder_output["y"].try_extract_array::<i64>().unwrap().to_owned();
                    for i in 0..num_layers {
                        k_caches[i] = s_decoder_output[format!("k_cache_{}", i)].try_extract_array::<f32>().unwrap().to_owned();
                        v_caches[i] = s_decoder_output[format!("v_cache_{}", i)].try_extract_array::<f32>().unwrap().to_owned();
                    }
                    y_emb = s_decoder_output["y_emb"].try_extract_array::<f32>().unwrap().to_owned();
                    let logits = s_decoder_output["logits"].try_extract_array::<f32>().unwrap().to_owned();
                    let samples = s_decoder_output["samples"].try_extract_array::<i32>().unwrap().to_owned();

                    debug!("S Decoder iteration {}: y shape = {:?}", idx, y.shape());
                    if idx >= 1500
                        || (y.shape()[1] - prefix_len) > EARLY_STOP_NUM
                        || argmax(&logits.view()).1 == T2S_DECODER_EOS
                        || samples
                            .get((0, 0).into_dimension())
                            .unwrap_or(&(T2S_DECODER_EOS as i32))
                            == &(T2S_DECODER_EOS as i32)
                    {
                        let mut y = y.to_owned();
                        if let Some(last) = y.last_mut() {
                            *last = 0;
                        }
                        break y
                            .slice(s![.., y.shape()[1] - idx..; 1])
                            .into_owned()
                            .insert_axis(Axis(0));
                    }

                    idx += 1;
                };
                info!("T2S S Decoder time: {:?}", time.elapsed().unwrap());

                // Final SoVITS processing
                let time = SystemTime::now();
                let outputs = match self.sovits.run(ort::inputs![
                    "text_seq" => Tensor::from_array(text_seq).unwrap(),
                    "pred_semantic" => Tensor::from_array(pred_semantic).unwrap(),
                    "ref_audio" => Tensor::from_array(ref_data.ref_audio_32k.to_owned()).unwrap()
                ]) {
                    Ok(output) => output,
                    Err(e) => {
                        error!("SoVITS failed: {}", e);
                        yield Err(GSVError::from(format!("SoVITS failed: {}", e)));
                        return;
                    }
                };
                let output = outputs["audio"].try_extract_array::<f32>().unwrap();
                info!("SoVITS time: {:?}", time.elapsed().unwrap());

                // Yield audio samples
                let samples = output.into_owned().into_raw_vec();
                debug!("Yielding {} samples for normalized text", samples.len());
                for sample in samples {
                    yield Ok(sample * 4.0);
                }
            }
        };

        Ok((spec, Box::pin(stream)))
    }

    // Synchronous wrapper for run
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

// Simple resampling function (linear interpolation)
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
