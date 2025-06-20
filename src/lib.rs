// src/lib.rs
use hound::{WavReader, WavSpec};
use jieba_rs::Jieba;
use log::{debug, info};
use ndarray::{Array, Dim, IxDyn};
// use ort::{
//     execution_providers::{CPUExecutionProvider, XNNPACKExecutionProvider},
//     session::{Session, builder::GraphOptimizationLevel},
// };
use std::{fs::File, sync::Arc};
use std::path::Path;
use std::{num::NonZero, str::FromStr};

mod error;
mod mnn_ffi;
mod text;
mod utils;

use crate::mnn_ffi::MNNModelWrapper;
use crate::text::TextProcessor;
use crate::text::g2pw::G2PWConverter;
pub use error::GSVError;

#[derive(Clone)]
pub struct ReferenceData {
    ref_seq: Array<i64, Dim<[usize; 2]>>,
    ref_bert: Array<f32, Dim<[usize; 2]>>,
    ref_audio_32k: Array<f32, Dim<[usize; 2]>>,
    ssl_content: Array<f32, IxDyn>,
}

pub struct TTSModel {
    text_processor: TextProcessor,
    mnn_model: MNNModelWrapper,
    ref_data: Option<ReferenceData>,
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
        info!("Initializing TTSModel with MNN sessions");

        // let g2pw_session = Session::builder()?
        //     .with_execution_providers([XNNPACKExecutionProvider::default()
        //         .with_intra_op_num_threads(NonZero::new(8).unwrap())
        //         .build()])?
        //     .with_optimization_level(GraphOptimizationLevel::Level3)?
        //     .commit_from_file(g2pw_path)?;

        let tokenizer =
            Arc::new(tokenizers::Tokenizer::from_str(text::g2pw::G2PW_TOKENIZER).unwrap());
        let text_processor = TextProcessor {
            jieba: Jieba::new(),
            g2pw: G2PWConverter::new().unwrap(),
            symbols: text::symbols::SYMBOLS.clone(),
        };

        let mnn_model = MNNModelWrapper::new(
            sovits_path.as_ref().to_str().unwrap(),
            ssl_path.as_ref().to_str().unwrap(),
            t2s_encoder_path.as_ref().to_str().unwrap(),
            t2s_fs_decoder_path.as_ref().to_str().unwrap(),
            t2s_s_decoder_path.as_ref().to_str().unwrap(),
            num_layers,
        )?;

        Ok(TTSModel {
            text_processor,
            mnn_model,
            ref_data: None,
        })
    }

    pub fn process_reference<P: AsRef<Path>>(
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
        let ref_seq = Array::from_shape_vec((1, ref_seq.len()), ref_seq)?;
        let ref_bert = Array::<f32, _>::zeros((ref_seq.shape()[1], 1024));

        let file = File::open(&reference_audio_path)?;
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
        let ref_audio_16k = Array::from_shape_vec((1, ref_audio_16k.len()), ref_audio_16k)?;

        let ref_audio_32k = resample_audio(audio_samples, spec.sample_rate, 32000)?;
        let ref_audio_32k = Array::from_shape_vec((1, ref_audio_32k.len()), ref_audio_32k)?;

        let ssl_content_vec = self.mnn_model.run_ssl(ref_audio_16k.as_slice().unwrap())?;
        let ssl_content = Array::from_shape_vec(vec![1, 768, ssl_content_vec.len()/768 ], ssl_content_vec)?;

        self.ref_data = Some(ReferenceData {
            ref_seq,
            ref_bert,
            ref_audio_32k,
            ssl_content,
        });

        Ok(())
    }

    pub fn run(&mut self, text: &str) -> Result<(WavSpec, Vec<f32>), GSVError> {
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

        let mut samples = Vec::new();
        for text_seq_vec in text_seqs {
            let text_seq = Array::from_shape_vec((1, text_seq_vec.len()), text_seq_vec)?;
            let text_bert = Array::<f32, _>::zeros((text_seq.shape()[1], 1024));

            let output_vec = self.mnn_model.run_inference(
                ref_data.ref_seq.as_slice().unwrap(),
                ref_data.ref_bert.as_slice().unwrap(),
                text_seq.as_slice().unwrap(),
                text_bert.as_slice().unwrap(),
                ref_data.ssl_content.as_slice().unwrap(),
                ref_data.ref_audio_32k.as_slice().unwrap(),
            )?;
            samples.extend(output_vec.into_iter().map(|s| s * 4.0));
        }

        Ok((spec, samples))
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
