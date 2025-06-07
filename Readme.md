# GPT-SOVITS-ONNX-RS

本项目旨在使用 **Rust** 与 **ONNX Runtime** 将 GPT-SoVITS（V2 版本）部署至任意支持 x86 或 ARM 架构的 CPU 设备上。
目前暂不支持 V3/V4 版本，原因在于其计算复杂度较高，尚难在通用 CPU 上实现令人满意的实时性能。

---

## 项目组件

本项目包含以下核心部分：

1. **Rust 推理运行时**：适用于构建 x86 平台的推理可执行程序。
2. **模型转换与优化脚本**：位于 `scripts` 目录，支持一键导出与优化 SoVITS 模型。
3. **Android 构建支持脚本**：由于 `ort-rs` 官方不提供 Android 的预构建包，需用户自行构建 ONNX Runtime。

本项目基于 [mzdk100/GPT-SoVITS](https://github.com/mzdk100/GPT-SoVITS) 开发和改进，目标是在中高端 x86/ARM CPU 上，通过 ONNX Runtime 实现可接受延迟的 TTS 体验。

---

## 项目目标

项目将作为个人全平台chatbot（尚未开源）的TTS引擎，为在Android侧实现低延迟、高可用的TTS能力。后续将围绕实时性进行优化（可能牺牲更大的精度）。

如果后续出现了更加高效的实现（比如如果MNN/ExecuTorch的版本切更快），本项目将停止开发。

---

## 方案对比（当前情况）

| 方案                                                                   | TTS 效果                 | 性能                     | 平台兼容性              | 易用性                         |
| -------------------------------------------------------------------- | ---------------------- | ---------------------- | ------------------ | --------------------------- |
| **sherpa-onnx**                                                      | ★★★☆☆（情感表达） | ★★★★★（模型小，可实时）    | ★★★★★（跨平台）         | ★★★★★（全平台预构建）             |
| **[GPT-SoVITS-Rust](https://github.com/second-state/gpt_sovits_rs)** | ★★★★★（接近原版效果）     | ★★★★☆（Torch）      | ★★☆☆☆（不支持 Android） | ★★★☆☆（需手动）              |
| **本项目（GPT-SOVITS-ONNX-RS）**                                          | ★★★★☆（与原版有差距，不稳定）            | ★★★☆☆（ONNX） | ★★★★☆（支持 ARM/x86）  | ★★★★☆（Android 需手动） |

---

## 使用建议（当前情况）

* ✅ **Android 平台 + 高性能需求**：推荐使用 `sherpa-onnx`
* ✅ **x86 平台（Linux/Windows, CUDA/CPU）+ 高拟真度需求**：推荐 [`GPT-SoVITS-Rust`](https://github.com/second-state/gpt_sovits_rs)
* ✅ **Android 平台（CPU）+ 高保真需求**：可以使用本项目（GPT-SOVITS-ONNX-RS）

---

## 构建指南

### x86 平台构建

使用 Cargo 编译：

```bash
cargo build --release
```

---

### Android 平台构建

> ⚠️ 为保持 ONNX Runtime 的更新与兼容性，**本项目不提供预构建二进制文件**，需用户自行构建。

构建步骤如下：

1. 安装 CMake ≥ 3.28（建议使用 Conda 环境安装以绕过旧系统限制）。
2. 下载并配置 Android NDK 与 SDK，设置必要的环境变量。
3. 执行 `onnx/build_android` 目录下的脚本，构建 ONNX Runtime。
4. 运行 `build_for_android.sh`，生成 Android 平台下的可执行程序和动态库。

---

## 实验性：其他 EP / Runtime 支持

借助 ONNX 的良好兼容性，你可以尝试使用不同的执行后端（Execution Provider）或运行时引擎运行模型。以下为当前已测试的组合：

### Execution Provider（EP）

* ⚠️ **NNAPI**：可运行，但未观察到明显性能提升。
* 🚧 **XNNPACK**：当前模型结构不支持。
* ❓ **QNN**： 理论上QNN的HTP后端可以提供更快的性能，后续探索。

### ONNX 替代 Runtime

* ⚠️ **MNN**：模型可转换，但现有 Rust 封装（wrapper）尚不成熟，文档缺失且接口不稳定，尚未能稳定运行推理流程。

---

如有问题或建议，欢迎通过 Issue 反馈。欢迎贡献代码或改进方案！
