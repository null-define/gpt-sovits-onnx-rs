use futures::StreamExt;
use gpt_sovits_onnx_rs::*;
use hound::WavWriter;
use std::path::Path;
use std::time::Instant;
use tokio::runtime::Runtime;

// Function to calculate statistics
fn calculate_stats(times: &[f64]) -> (f64, f64, f64, f64) {
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
    let max = sorted.last().unwrap_or(&0.0);
    let min = sorted.first().unwrap_or(&0.0);
    (avg, median, *max, *min)
}

fn main() -> Result<(), GSVError> {
    env_logger::init();
    let assets_dir = Path::new("/home/qiang/projects/GPT-SoVITS/onnx-patched/kaoyu");

    // Initialize model once
    let mut model = TTSModel::new(
        assets_dir.join("g2pW.onnx"),
        assets_dir.join("kaoyu_vits.onnx"),
        assets_dir.join("kaoyu_ssl.onnx"),
        assets_dir.join("kaoyu_t2s_encoder.onnx"),
        assets_dir.join("kaoyu_t2s_fs_decoder.onnx"),
        assets_dir.join("kaoyu_t2s_s_decoder.onnx"),
        24,
        Some(assets_dir.join("kaoyu_bert.onnx")),
    )?;

    // Process reference audio and text synchronously
    model.process_reference_sync(
        assets_dir.join("ref.wav"),
        "格式化，可以给自家的奶带来大量的。",
    )?;

    const NUM_RUNS: usize = 1;
    let text = "小鱼想成为你的好朋友，而不仅仅是一个可爱的AI助理。我想成为一个足够interesting的AI assistant，你呢？";

    // Synchronous inference timing
    let mut sync_times = Vec::new();
    for i in 0..NUM_RUNS {
        let start = Instant::now();
        let (spec, samples) = model.run_sync(text)?;
        let duration = start.elapsed().as_secs_f64() * 1000.0; // Convert to milliseconds
        sync_times.push(duration);

        // Write to WAV file only for the last run to avoid overwriting
        if i == NUM_RUNS - 1 {
            let mut writer = WavWriter::create("output_sync.wav", spec)?;
            for sample in samples {
                writer.write_sample(sample)?;
            }
            writer.finalize()?;
        }
    }

    // Calculate statistics for sync
    let (sync_avg, sync_median, sync_max, sync_min) = calculate_stats(&sync_times);
    println!("Synchronous Inference ({} runs):", NUM_RUNS);
    println!("  Average: {:.2} ms", sync_avg);
    println!("  Median: {:.2} ms", sync_median);
    println!("  Max: {:.2} ms", sync_max);
    println!("  Min: {:.2} ms", sync_min);

    // Asynchronous inference timing
    let rt = Runtime::new()?;
    let mut async_times = Vec::new();
    let async_text = "你好呀，我们是一群追逐梦想的人！";
    rt.block_on(async {
        for i in 0..NUM_RUNS {
            let start = Instant::now();
            let (spec, stream) = model.run(async_text).await?;
            let mut writer = if i == NUM_RUNS - 1 {
                Some(WavWriter::create("output_async.wav", spec)?)
            } else {
                None
            };
            futures::pin_mut!(stream);
            while let Some(sample) = stream.next().await {
                if let Some(ref mut w) = writer {
                    w.write_sample(sample?)?;
                }
            }
            if let Some(w) = writer {
                w.finalize()?;
            }
            let duration = start.elapsed().as_secs_f64() * 1000.0; // Convert to milliseconds
            async_times.push(duration);
        }
        Ok::<(), GSVError>(())
    })?;

    // Calculate statistics for async
    let (async_avg, async_median, async_max, async_min) = calculate_stats(&async_times);
    println!("Asynchronous Inference ({} runs):", NUM_RUNS);
    println!("  Average: {:.2} ms", async_avg);
    println!("  Median: {:.2} ms", async_median);
    println!("  Max: {:.2} ms", async_max);
    println!("  Min: {:.2} ms", async_min);

    Ok(())
}