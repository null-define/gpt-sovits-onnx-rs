use futures::StreamExt;
use gpt_sovits_onnx_rs::*;
use hound::WavWriter;
use std::path::Path;
use tokio::runtime::Runtime;

// Example usage
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
    )?;

    // Process reference audio and text synchronously
    model.process_reference_sync(
        assets_dir.join("ref.wav"),
        "格式化，可以给自家的奶带来大量的",
    )?;

    // Run inference synchronously
    let (spec, samples) = model.run_sync("Hello, this is a test.")?;

    // Write to WAV file
    {
        let mut writer = WavWriter::create("output.wav", spec)?;
        for sample in samples {
            writer.write_sample(sample)?;
        }
        writer.finalize()?;
    }

    // Example async usage (for comparison)
    let rt = Runtime::new()?;
    rt.block_on(async {
        let (spec, stream) = model
            .run("你好呀，我们是一群追逐梦想的人！")
            // .run("今天天气很不错，天空全都是乌云，所以心里不开心。想要吃点好的犒劳一下自己。")
            .await?;
        let mut writer = WavWriter::create("output_async.wav", spec)?;
        futures::pin_mut!(stream);
        while let Some(sample) = stream.next().await {
            writer.write_sample(sample?)?;
        }
        writer.finalize()?;
        Ok::<(), GSVError>(())
    })?;

    Ok(())
}
