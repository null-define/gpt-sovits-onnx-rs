# GPT-SOVITS-ONNX-RS

A lightweight, cross-platform GPT-SoVITS TTS inference engine based on **Rust** and **ONNX Runtime**, designed specifically to run on **x86/ARM** architecture **CPUs**.

> ⚠️ Important Note
>
> mainly focus on Chinese TTS, English quality is not guaranteed.

## Project Introduction

This project aims to deploy the GPT-SoVITS (V2) model to various CPU devices using ONNX Runtime, achieving low-latency, high-availability local Text-to-Speech (TTS) capabilities. It was initially developed to provide TTS support for the Android and PC clients of a personal, all-platform Chatbot project (not yet open-sourced).

Continuous optimization for real-time performance will be pursued while ensuring acceptable accuracy.

> **Version Support Note**
>
> Currently, only **GPT-SoVITS V2** is supported. V3/V4 versions are not in the support plan for now, as their higher computational complexity makes it difficult to achieve satisfactory real-time inference performance on general-purpose CPUs.

> **Future of the Project**
>
> The goal is to create an efficient CPU inference solution. If a more efficient implementation emerges in the community in the future (e.g., a solution based on MNN with better performance), this project will have fulfilled its exploratory mission and will cease active development to avoid redundant work.

-----

## Core Features

* **Cross-Platform Inference Core**: Written in Rust to ensure memory safety and high performance, easily compilable for x86 and ARM platforms (Linux, Android, etc.).
* **One-Click Model Conversion**: Provides Python scripts in the `scripts` directory for one-click export and optimization of the SoVITS model. **Note:** The optimized model structure is not compatible with the official one. You must use the complete scripts provided by this project and follow the documentation during conversion.
* **Complete Android Build Support**: Includes a `build_for_android.sh` script that automates the source code download and compilation of ONNX Runtime and the project build, addressing the issue of official `ort-rs` lacking pre-built packages for Android.

-----

## Project Status & Known Issues

* **Poor Performance with English and Mixed Chinese-English Input**: Currently borrows some logic from [GPT-SoVITS-RS](https://github.com/second-state/gpt_sovits_rs), using G2pw to preprocess Chinese. This has improved the results, but there is still room for enhancement.
* **Insufficient Output Stability**: Occasional issues like premature audio truncation exist and will be gradually fixed in future versions.
* **Performance is Still Being Optimized**: For detailed performance data, please refer to the [**Performance Records (perf\_record)**](https://www.google.com/search?q=doc/perf_record.md).

-----

## Demo Showcase

A demonstration was conducted on an Android device to visually present the current results.

* **Source Code**: [gpt-sovits-android-demo](https://github.com/null-define/gpt-sovits-android-demo/tree/master)
* **Demo Video**:
    [https://github.com/user-attachments/assets/369cefb6-3dab-4db4-9bb1-647f526f27d0](https://github.com/user-attachments/assets/369cefb6-3dab-4db4-9bb1-647f526f27d0)

> **Note**: The demo device is an iQOO 13. Actual inference times may vary significantly across different SoCs and devices.

-----

## Solution Comparison

To help you choose the most suitable solution, I have compared it with mainstream community projects.

| Solution | TTS Quality | Performance | Platform Compatibility | Ease of Use |
| :--- | :--- | :--- | :--- | :--- |
| **sherpa-onnx** | ★★★☆☆ (Slightly weaker emotion) | ★★★★★ (Small model, strong real-time) | ★★★★★ (All platforms) | ★★★★★ (Official pre-builts) |
| **[GPT-SoVITS-RS](https://github.com/second-state/gpt_sovits_rs)** | ★★★★★ (Near original quality) | ★★★★☆ (Depends on Torch) | ★★☆☆☆ (Poor Android support) | ★★★☆☆ (Manual configuration) |
| **This Project** | ★★★☆☆ (Currently unstable) | ★★★★☆ (ONNX optimized) | ★★★☆☆ (Supports ARM/x86) | ★★★★☆ (Requires execute build script for Android) |

-----

## Usage Recommendations

Based on your specific needs, the following are recommended:

* **For pursuing maximum performance and ease of use on the Android platform**:
  * ✅ **`sherpa-onnx` is recommended**
* **For pursuing high-fidelity results on x86 platforms (Linux/Windows, CUDA/CPU)**:
  * ✅ **`GPT-SoVITS-RS` is recommended**
* **For pursuing high-fidelity results and needing to run on both Android and x86 CPUs**:
  * ✅ **You can try this project** and are welcome to help improve it\!

-----

## Model Download

If you don't want to train and export the model yourself, you can use a pre-trained model for a quick start.

* **Download Link**: [huggingface.co/mikv39/gpt-sovits-onnx-custom](https://huggingface.co/mikv39/gpt-sovits-onnx-custom)
* This model can be directly loaded and used in the [gpt-sovits-android-demo](https://github.com/null-define/gpt-sovits-android-demo/tree/master).

> **Copyright Notice**: This model was fine-tuned using copyrighted audio and video materials. Please do not use it for any commercial purposes.

-----

## Build Guide

### 1\. Model Conversion

Please refer to the documentation in the `scripts` directory: [scripts/README.md](https://www.google.com/search?q=scripts/README.md).

### 2\. Build for x86 Platforms (Linux/Windows/macOS)

You can compile directly using Cargo:

```bash
cargo build --release
```

### 3\. Build for Android Platform

> ⚠️ **Important Note**
>
> To ensure flexibility and timely updates of the ONNX Runtime version, **this project does not provide pre-built binaries**. You need to build it yourself following the steps below.

1. **Environment Setup**:

      * Install CMake ≥ 3.28 (using Conda is recommended to avoid system version limitations).
      * Download and configure the Android NDK and SDK, and set the relevant environment variables.

2. **First-time Build**:

      * Run the one-click script. It will automatically download the ONNX Runtime source code, compile it, and build the executable and dynamic library for Android.

    <!-- end list -->

    ```bash
    ./build_for_android.sh
    ```

3. **Subsequent Incremental Builds**:

      * If you only modify the Rust code, you can compile directly using the Cargo command.

    <!-- end list -->

    ```bash
    cargo build --target aarch64-linux-android --release --features jni --examples
    ```

-----

## Exploration of Experimental Features

I have conducted preliminary tests using other Execution Providers (EP) or alternative runtimes.

### ONNX Execution Provider (EP)

* ✅ **XNNPACK**: Can be used to accelerate the Decoder model, but no significant performance improvement was observed on the test device (iQOO 13). This conclusion may not apply to all hardware platforms.
* ⚠️ **NNAPI**: All models can run, but no performance improvement was seen, regardless of whether fp16 or fp32 was used. Google no longer recommends prioritizing NNAPI.

### ONNX Alternative Runtimes

* ❌ **MNN**: Attempted to convert the ONNX model to the MNN format. Although the model could be successfully loaded using the MNN C++ API (v3.2.0), the Decoder part encountered memory allocation failures or missing input errors at runtime (the specific error depended on whether the optimization script was used). As this may be a bug in MNN itself and due to limited personal time and energy, the MNN solution has been temporarily shelved. Some code has been uploaded to the [mnn\_dev\_backup branch](https://github.com/null-define/gpt-sovits-onnx-rs/tree/mnn_dev_backup). Experts who are willing to explore can use this branch.
