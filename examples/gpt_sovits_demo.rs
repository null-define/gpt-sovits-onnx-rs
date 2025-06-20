use futures::StreamExt;
use gpt_sovits_onnx_rs::*;
use hound::WavWriter;
use std::path::Path;
use tokio::runtime::Runtime;

// Example usage
fn main() -> Result<(), GSVError> {
    env_logger::init();
    let assets_dir = Path::new("/home/qiang/projects/GPT-SoVITS/mnn");

    // Initialize model once
    let mut model = TTSModel::new(
        assets_dir.join("/home/qiang/projects/GPT-SoVITS/onnx-patched/kaoyu/g2pW.onnx"),
        assets_dir.join("kaoyu_vits.mnn"),
        assets_dir.join("kaoyu_ssl.mnn"),
        assets_dir.join("kaoyu_t2s_encoder.mnn"),
        assets_dir.join("kaoyu_t2s_fs_decoder.mnn"),
        assets_dir.join("kaoyu_t2s_s_decoder.mnn"),
        24,
    )?;
    println!("load success");

    // Process reference audio and text synchronously
    model.process_reference(
        assets_dir.join("ref.wav"),
        "格式化，可以给自家的奶带来大量的",
    )?;

    println!("run reference success");

    // Run inference synchronously
    let (spec, samples) = model.run("小鱼想成为你的好朋友，而不仅仅是一个可爱的AI助理")?;

    // Write to WAV file
    {
        let mut writer = WavWriter::create("output.wav", spec)?;
        for sample in samples {
            writer.write_sample(sample)?;
        }
        writer.finalize()?;
    }

    // Example async usage (for comparison)

    Ok(())
}
