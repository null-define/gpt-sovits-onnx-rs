use hound::{WavReader, WavSpec, WavWriter};
use ndarray::{Array, ArrayView, Axis, Dim, IntoDimension, IxDyn, s};
use ort::{
    execution_providers::CPUExecutionProvider,
    session::Session,
    value::{Tensor, TensorRef},
};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
    time::SystemTime,
};
use futures::{stream::Stream, StreamExt};
use async_stream::stream;
use tokio::runtime::Runtime;

// Module imports (assuming these are in separate files)
mod error;
mod phoneme;

pub use {
    error::GSVError, // Explicitly import the error type
    phoneme::*,
};

const EARLY_STOP_NUM: usize = 2700;
const T2S_DECODER_EOS: usize = 1024;

fn argmax(tensor: &ArrayView<f32, IxDyn>) -> (usize, usize) {
    let mut max_index = (0, 0);
    let mut max_value = tensor
        .get(IxDyn::zeros(2))
        .copied()
        .unwrap_or(f32::NEG_INFINITY);

    for i in 0..tensor.shape()[0] {
        for j in 0..tensor.shape()[1] {
            if let Some(value) = tensor.get((i, j).into_dimension()) {
                if *value > max_value {
                    max_value = *value;
                    max_index = (i, j);
                }
            }
        }
    }
    max_index
}

// Struct to hold model sessions and reusable reference data
pub struct TTSModel {
    sovits: Session,
    ssl: Session,
    t2s_encoder: Session,
    t2s_fs_decoder: Session,
    t2s_s_decoder: Session,
    ref_data: Option<ReferenceData>,
}

pub struct ReferenceData {
    ref_seq: Array<i64, Dim<[usize; 2]>>,
    ref_bert: Array<f32, Dim<[usize; 2]>>,
    ref_audio_32k: Array<f32, Dim<[usize; 2]>>,
    ssl_content: Array<f32, IxDyn>,
}

impl TTSModel {
    // Initialize the model with ONNX sessions
    pub fn new<P: AsRef<Path>>(
        sovits_path: P,
        ssl_path: P,
        t2s_encoder_path: P,
        t2s_fs_decoder_path: P,
        t2s_s_decoder_path: P,
    ) -> Result<Self, GSVError> {
        let sovits = Session::builder()?
            .with_execution_providers([CPUExecutionProvider::default().build()])?
            .commit_from_file(sovits_path)?;
        let ssl = Session::builder()?
            .with_execution_providers([CPUExecutionProvider::default().build()])?
            .commit_from_file(ssl_path)?;
        let t2s_encoder = Session::builder()?
            .with_execution_providers([CPUExecutionProvider::default().build()])?
            .commit_from_file(t2s_encoder_path)?;
        let t2s_fs_decoder = Session::builder()?
            .with_execution_providers([CPUExecutionProvider::default().build()])?
            .commit_from_file(t2s_fs_decoder_path)?;
        let t2s_s_decoder = Session::builder()?
            .with_execution_providers([CPUExecutionProvider::default().build()])?
            .commit_from_file(t2s_s_decoder_path)?;

        Ok(TTSModel {
            sovits,
            ssl,
            t2s_encoder,
            t2s_fs_decoder,
            t2s_s_decoder,
            ref_data: None,
        })
    }

    // Process reference audio and text once (async)
    pub async fn process_reference<P: AsRef<Path>>(
        &mut self,
        reference_audio_path: P,
        ref_text: &str,
    ) -> Result<(), GSVError> {
        // Text processing
        let mut extractor = PhonemeExtractor::default();
        extractor.push_str(ref_text);
        let ref_seq = extractor.get_phone_ids();
        println!("Reference phoneme extractor: {:?}", extractor);
        let ref_seq = Array::from_shape_vec((1, ref_seq.len()), ref_seq)?;
        let ref_bert = Array::<f32, _>::zeros((ref_seq.shape()[1], 1024));

        // Read and resample reference audio
        let mut file = File::open(&reference_audio_path)?;
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

        // SSL processing
        let time = SystemTime::now();
        let ssl_output = self.ssl.run(ort::inputs![
            "ref_audio_16k" => Tensor::from_array(ref_audio_16k.clone())?
        ])?;
        println!("SSL Cost is {:?}", time.elapsed()?);
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
        let rt = Runtime::new()?;
        rt.block_on(self.process_reference(reference_audio_path, ref_text))
    }

    // Run inference with given text and return a stream (async)
    pub async fn run(
        &mut self,
        text: &str,
    ) -> Result<(WavSpec, impl Stream<Item = Result<f32, GSVError>> + Send + Unpin), GSVError> {
        let ref_data = self
            .ref_data
            .as_ref()
            .ok_or_else(|| GSVError::from("Reference data not initialized"))?;

        // Text processing
        let mut extractor = PhonemeExtractor::default();
        extractor.push_str(text);
        let text_seq = extractor.get_phone_ids();
        println!("Text phoneme extractor: {:?}", extractor);
        let text_seq = Array::from_shape_vec((1, text_seq.len()), text_seq)?;
        let text_bert = Array::<f32, _>::zeros((text_seq.shape()[1], 1024));

        // T2S Encoder
        let time = SystemTime::now();
        let encoder_output = self.t2s_encoder.run(ort::inputs![
            "ref_seq" => Tensor::from_array(ref_data.ref_seq.to_owned())?,
            "text_seq" => Tensor::from_array(text_seq.to_owned())?,
            "ref_bert" => Tensor::from_array(ref_data.ref_bert.to_owned())?,
            "text_bert" => Tensor::from_array(text_bert.to_owned())?,
            "ssl_content" => Tensor::from_array(ref_data.ssl_content.to_owned())?,
        ])?;
        println!("T2S Encoder Cost is {:?}", time.elapsed()?);
        let x = encoder_output["x"].try_extract_array::<f32>()?.to_owned();
        let prompts = encoder_output["prompts"]
            .try_extract_array::<i64>()?
            .to_owned();
        let prefix_len = prompts.dim()[1];

        // T2S FS Decoder
        let time = SystemTime::now();
        let fs_decoder_output = self.t2s_fs_decoder.run(ort::inputs![
            "x" => Tensor::from_array(x.to_owned())?,
            "prompts" => Tensor::from_array(prompts.to_owned())?
        ])?;
        println!("T2S FS Decoder Cost is {:?}", time.elapsed()?);
        let (mut y, mut k, mut v, mut y_emb, x_example) = (
            fs_decoder_output["y"]
                .try_extract_array::<i64>()?
                .to_owned(),
            fs_decoder_output["k"]
                .try_extract_array::<f32>()?
                .to_owned(),
            fs_decoder_output["v"]
                .try_extract_array::<f32>()?
                .to_owned(),
            fs_decoder_output["y_emb"]
                .try_extract_array::<f32>()?
                .to_owned(),
            fs_decoder_output["x_example"]
                .try_extract_array::<f32>()?
                .to_owned(),
        );

        // T2S S Decoder
        let time = SystemTime::now();
        let mut idx = 1;
        let pred_semantic = loop {
            let time = SystemTime::now();
            let s_decoder_output = self.t2s_s_decoder.run(ort::inputs![
                "iy" => Tensor::from_array(y.to_owned())?,
                "ik" => Tensor::from_array(k.to_owned())?,
                "iv" => Tensor::from_array(v.to_owned())?,
                "iy_emb" => Tensor::from_array(y_emb.to_owned())?,
                "ix_example" => Tensor::from_array(x_example.to_owned())?
            ])?;
            println!("T2S S Decoder Forward Once Cost is {:?}", time.elapsed()?);

            let (y_new, k_new, v_new, y_emb_new, samples, logits) = (
                s_decoder_output["y"].try_extract_array::<i64>()?.to_owned(),
                s_decoder_output["k"].try_extract_array::<f32>()?.to_owned(),
                s_decoder_output["v"].try_extract_array::<f32>()?.to_owned(),
                s_decoder_output["y_emb"]
                    .try_extract_array::<f32>()?
                    .to_owned(),
                s_decoder_output["samples"]
                    .try_extract_array::<i32>()?
                    .to_owned(),
                s_decoder_output["logits"]
                    .try_extract_array::<f32>()?
                    .to_owned(),
            );

            y = y_new;
            k = k_new;
            v = v_new;
            y_emb = y_emb_new;

            println!("y: {:?}", y.last());
            println!("y shape: {:?}", y.shape());
            println!("idx: {:?}", idx);

            if idx >= 1500
                || (y.dim()[1] - prefix_len) > EARLY_STOP_NUM.try_into().unwrap()
                || argmax(&logits.view()).1 == T2S_DECODER_EOS
                || samples
                    .get((0, 0).into_dimension())
                    .unwrap_or(&(T2S_DECODER_EOS as i32))
                    == &(T2S_DECODER_EOS as i32)
            {
                let mut y = y.to_owned();
                y.last_mut().and_then(|i| {
                    *i = 0;
                    None::<()>
                });

                break y
                    .slice(s![.., y.shape()[1] - idx..; 1])
                    .into_owned()
                    .insert_axis(Axis(0));
            }

            idx += 1;
            println!(
                "T2S S Decoder Forward Once Other Cost is {:?}",
                time.elapsed()?
            );
        };
        println!("T2S S Decoder Cost is {:?}", time.elapsed()?);

        // Final SoVITS processing
        let time = SystemTime::now();
        let outputs = self.sovits.run(ort::inputs![
            "text_seq" => Tensor::from_array(text_seq.to_owned())?,
            "pred_semantic" => Tensor::from_array(pred_semantic.to_owned())?,
            "ref_audio" => Tensor::from_array(ref_data.ref_audio_32k.to_owned())?
        ])?;
        let output = outputs["audio"].try_extract_array::<f32>()?;
        println!("SoVITS Cost is {:?}", time.elapsed()?);

        // Define WAV specification
        let spec = WavSpec {
            channels: 1,
            sample_rate: 32000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        // Create a stream for the audio samples
        let samples = output.as_slice().unwrap().to_vec();
        let stream = stream! {
            for sample in samples {
                yield Ok(sample);
            }
        };

        // Pin the stream to make it Unpin
        Ok((spec, Box::pin(stream)))
    }

    // Synchronous wrapper for run
    pub fn run_sync(&mut self, text: &str) -> Result<(WavSpec, Vec<f32>), GSVError> {
        let rt = Runtime::new()?;
        rt.block_on(async {
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

// Example usage
// fn main() -> Result<(), GSVError> {
//     // Initialize model once
//     let mut model = TTSModel::new(
//         "path/to/sovits.onnx",
//         "path/to/ssl.onnx",
//         "path/to/t2s_encoder.onnx",
//         "path/to/t2s_fs_decoder.onnx",
//         "path/to/t2s_s_decoder.onnx",
//     )?;

//     // Process reference audio and text synchronously
//     model.process_reference_sync("path/to/ref_audio.wav", "reference text")?;

//     // Run inference synchronously
//     let (spec, samples) = model.run_sync("Hello, this is a test.")?;

//     // Write to WAV file
//     let mut writer = WavWriter::create("output.wav", spec)?;
//     for sample in samples {
//         writer.write_sample(sample)?;
//     }
//     writer.finalize()?;

//     // Example async usage (for comparison)
//     let rt = Runtime::new()?;
//     rt.block_on(async {
//         model
//             .process_reference("path/to/ref_audio.wav", "reference text")
//             .await?;
//         let (spec, stream) = model.run("Another test.").await?;
//         let mut writer = WavWriter::create("output_async.wav", spec)?;
//         futures::pin_mut!(stream);
//         while let Some(sample) = stream.next().await {
//             writer.write_sample(sample?)?;
//         }
//         writer.finalize()?;
//         Ok::<(), GSVError>(())
//     })?;

//     Ok(())
// }