# GPT-SOVITS-ONNX-RS

一个基于 **Rust** 和 **ONNX Runtime** 的轻量级、跨平台 GPT-SoVITS TTS 推理引擎，专为在 **x86/ARM** 架构的 **CPU** 上运行而设计。

## 项目简介

本项目旨在将 GPT-SoVITS (V2) 模型通过 ONNX Runtime 部署到各类 CPU 设备上，以实现低延迟、高可用的本地文本转语音（TTS）能力。它最初是为个人全平台 Chatbot 项目（尚未开源）的 Android 和 PC 端提供 TTS 支持而开发的。

将在保证可接受精度的前提下，对实时性进行持续优化。

-----

## 核心特性

* **跨平台推理核心**：使用 Rust 编写，确保了内存安全与高性能，可轻松编译到 x86 和 ARM 平台（Linux, Android 等）。
* **一键式模型转换**：提供位于 `scripts` 目录的 Python 脚本，用于一键导出和优化 SoVITS 模型。**注意：** 优化后的模型结构与官方不兼容，转换时需使用本项目提供的完整脚本并按照文档操作。
* **完整的 Android 构建支持**：提供了 `build_for_android.sh` 脚本，自动化处理 ONNX Runtime 的源码下载、编译及项目构建，解决了官方 `ort-rs` 缺少 Android 预构建包的问题。

-----

## 项目状态与已知问题

2025-07-12: 将部分onnx模型代码同步到和pytorch一致，简化了一部分模型逻辑，并修复了library中由于缺少空白声音导致的吸气问题，如果提示找不到输出，请更新hf上的新模型，或者重新转换自己的模型。

2025-06-30: 提取sampler到rust层，并更新了模型转换脚本和demo模型，如果提示找不到输出，请更新模型。

* **性能持续优化中**：详细性能数据请参考 [**性能记录 (perf\_record)**](doc/perf_record.md)。
* **尝试引入粤语支持**: 通过在运行时设置lang_id为LangId::AutoYue，可以启用粤语模式。

-----

## Demo 展示

在 Android 设备上进行了演示，以直观展示当前效果。

* **源代码**: [gpt-sovits-android-demo](https://github.com/null-define/gpt-sovits-android-demo/tree/master)
* **演示视频**:

目前推理耗时在8 elite这一类的高端移动端soc上是可以接受的。

https://github.com/user-attachments/assets/03565b5f-a3e4-4689-83c5-7ed72b6acf86

当和LLM结合时，TTS的效果如下（此demo由于bug太多功能不稳定，尚未开源）

https://github.com/user-attachments/assets/158fafd4-e2c9-416d-92c8-8e4340220de8



> **注意**：演示机型为 iQOO 13。实际推理时间在不同 SoC 和设备上可能存在显著差异。

-----

## 方案对比

为了帮助您选择最适合的方案，我将其与社区主流项目进行了对比。

| 方案 | TTS 效果 | 性能 | 平台兼容性 | 易用性 |
| :--- | :--- | :--- | :--- | :--- |
| **sherpa-onnx** | ★★★☆☆ (情感稍弱) | ★★★★★ (模型小，实时性强) | ★★★★★ (全平台) | ★★★★★ (官方预构建) |
| **[GPT-SoVITS-RS](https://github.com/second-state/gpt_sovits_rs)** | ★★★★★ (接近原版) | ★★★★☆ (依赖 Torch) | ★★☆☆☆ (Android 支持不佳) | ★★★☆☆ (需手动配置) |
| **本项目** | ★★★☆☆ (当前不稳定) | ★★★★☆ (ONNX 优化) | ★★★☆☆ (支持 ARM/x86) | ★★★★☆ (Android 需手动执行构建脚本) |

-----

## 使用建议

根据您的具体需求，推荐如下：

* **追求极致性能和易用性的 Android 平台**：
  * ✅ **推荐使用 `sherpa-onnx`**
* **追求高拟真度且在 x86 平台（Linux/Windows, CUDA/CPU）**：
  * ✅ **推荐使用 `GPT-SoVITS-RS`**
* **追求高拟真度且需要在 Android 和 x86 CPU 上运行**：
  * ✅ **可以尝试本项目**，并欢迎帮助改进！

-----

## 模型下载

如果您不想自行训练和导出模型，可以使用预训练模型进行快速体验。

* **主模型下载地址**：[huggingface.co/mikv39/gpt-sovits-onnx-custom](https://huggingface.co/mikv39/gpt-sovits-onnx-custom)
* 该模型可直接在 [gpt-sovits-android-demo](https://github.com/null-define/gpt-sovits-android-demo/tree/master) 中加载使用，或者直接替换examples下的gpt_sovits_demo中的模型地址。

> **版权声明**：此模型使用了受版权保护的音视频素材进行微调，请勿用于任何商业用途。

**gp2en模型下载** 建议下载，参见[cisco-ai/mini-bart-g2p](https://huggingface.co/cisco-ai/mini-bart-g2p/tree/main/onnx),下载完成后可以把模型目录文件夹设置为TTSModel的g2p_en_path参数，启用gp2 en模型支持。默认的demo和JNI都启用了gp2 en模型，需要在原来的目录下新建一个g2p_en文件夹，把下载的模型放进去。


-----

## 构建指南

### 1\. 模型转换

请参考 `scripts` 目录下的说明文档：[scripts/README.md](scripts/README.md)。

### 2\. x86 平台构建 (Linux/Windows/macOS)

直接使用 Cargo 即可完成编译：

```bash
cargo build --release
```

### 3\. Android 平台构建

> ⚠️ **重要提示**
>
> 为确保 ONNX Runtime 版本的灵活性与及时更新，**本项目不提供预构建的二进制文件**。您需要根据以下步骤自行构建。

1. **环境准备**:
      * 安装 CMake ≥ 3.28 (推荐使用 Conda 安装以避免系统版本限制)。
      * 下载并配置 Android NDK 与 SDK，并设置好相关环境变量。
      * 在~/.cargo/config.toml中设置好`[target.aarch64-linux-android]`的linker和ar,注意androidN-clang的N最好>=28，
        * 最好保证build_for_android.sh中的android_api参数一致
        * 如果是较新版本的android系统，请使用16k page size `rustflags = ["-C", "link-arg=-Wl,-z,max-page-size=16384"]`

      ```toml
      [target.aarch64-linux-android]
      linker = "/android-ndk-r27c//toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android32-clang"
      ar = "/android-ndk-r27c//toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar"
      rustflags = ["-C", "link-arg=-Wl,-z,max-page-size=16384"]
      ```

2. **首次构建**:
      * 运行一键式脚本，该脚本将自动完成 ONNX Runtime 源码下载、编译，并构建适用于 Android 的可执行文件和动态库。
    <!-- end list -->
    ```bash
    ./build_for_android.sh
    ```

3. **后续增量构建**:
      * 如果仅修改了 Rust 代码，可直接使用 Cargo 命令进行编译。
    <!-- end list -->
    ```bash
    cargo build --target aarch64-linux-android --release --features jni --examples
    ```

-----

## 实验性功能探索

我对使用其他执行后端（Execution Provider）或替代运行时进行了初步测试。

### ONNX Execution Provider (EP)

* ✅ **XNNPACK**: 可用于加速 Decoder 模型，但在测试机（iQOO 13）上未观察到显著性能提升。此结论可能不适用于所有硬件平台。
* ⚠️ **NNAPI**: 所有模型均可运行，但无论使用 fp16 还是 fp32，均未带来性能改善。Google 官方已不推荐优先使用 NNAPI。

### ONNX 替代运行时

* ❌ **MNN**: 尝试将 ONNX 模型转换为 MNN 格式。虽然模型可以使用 MNN C++ API (v3.2.0) 成功加载，但在运行时，Decoder 部分出现内存分配失败或输入丢失的错误（具体错误取决于是否使用了优化脚本）。由于这可能是 MNN 本身的 Bug，且个人时间和精力有限，已暂时搁置 MNN 方案。部分代码已经上传到[mnn_dev_backup 分支](https://github.com/null-define/gpt-sovits-onnx-rs/tree/mnn_dev_backup)，如果有大佬愿意探索可以使用此分支，
