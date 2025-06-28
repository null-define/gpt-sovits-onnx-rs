use clap::Parser;
use futures::StreamExt;
use gpt_sovits_onnx_rs::*;
use hound::{WavSpec, WavWriter};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokio::runtime::Runtime;

#[derive(Parser, Debug)]
struct Args {
    #[arg(
        long,
        default_value = "/home/qiang/projects/GPT-SoVITS/onnx-patched/custom"
    )]
    model_path: PathBuf,
    #[arg(long, default_value_t = 1)]
    run_count: usize,
    #[arg(long, default_value = "喜欢我小鱼吗？小子！")]
    text: String,
    #[arg(long, default_value = "zh")] // can be zh/yue
    lang: String,
    #[arg(long, default_value = "格式化，可以给自家的奶带来大量的。")]
    ref_text: String,
}

struct TimingStats {
    avg: f64,
    median: f64,
    max: f64,
    min: f64,
}

impl TimingStats {
    fn new(times: &[f64]) -> Self {
        let count = times.len() as f64;
        let sum: f64 = times.iter().sum();
        let avg = sum / count;
        let mut sorted = times.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };
        Self {
            avg,
            median,
            max: *sorted.last().unwrap_or(&0.0),
            min: *sorted.first().unwrap_or(&0.0),
        }
    }

    fn print(&self, mode: &str, runs: usize) {
        println!("{} Inference ({} runs):", mode, runs);
        println!("  Average: {:.2} ms", self.avg);
        println!("  Median: {:.2} ms", self.median);
        println!("  Max: {:.2} ms", self.max);
        println!("  Min: {:.2} ms", self.min);
    }
}

fn create_model(assets_dir: &Path) -> Result<TTSModel, GSVError> {
    TTSModel::new(
        assets_dir.join("custom_vits.onnx"),
        assets_dir.join("ssl.onnx"),
        assets_dir.join("custom_t2s_encoder.onnx"),
        assets_dir.join("custom_t2s_fs_decoder.onnx"),
        assets_dir.join("custom_t2s_s_decoder.onnx"),
        24,
        Some(assets_dir.join("bert.onnx")),
        Some(assets_dir.join("g2pW.onnx")),
        Some(assets_dir.join("g2p_en")), // assume you have g2p en mode downloaded, can be none
    )
}

fn write_wav(spec: WavSpec, samples: &[f32], filename: &str) -> Result<(), GSVError> {
    let mut writer = WavWriter::create(filename, spec)?;
    for &sample in samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(())
}

fn run_sync_inference(
    model: &mut TTSModel,
    text: &str,
    lang: &str,
    runs: usize,
    output_file: &str,
) -> Result<TimingStats, GSVError> {
    let mut times = Vec::with_capacity(runs);
    let mut lang_id = LangId::Auto;
    if lang == "yue" {
        lang_id = LangId::AutoYue;
    }
    for i in 0..runs {
        let start = Instant::now();
        let (spec, samples) = model.synthesize_sync(text, lang_id)?;
        if i == runs - 1 {
            write_wav(spec, &samples, output_file)?;
        }
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(TimingStats::new(&times))
}

fn main() -> Result<(), GSVError> {
    env_logger::init();
    let args = Args::parse();

    let mut model = create_model(&args.model_path)?;
    model.process_reference_sync(
        args.model_path.join("ref.wav"),
        &args.ref_text,
        LangId::Auto,
    )?;

    let lang = args.lang;

    let stats = run_sync_inference(
        &mut model,
        &args.text,
        &lang,
        args.run_count,
        "output.wav",
    )?;
    stats.print("Synchronous", args.run_count);

    Ok(())
}
