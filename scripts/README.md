# 使用说明

这个目录下包含了所需要的转换脚本。需要和GPT_SoVITS原项目共同使用。


git clone原项目后，需要使用原项目的install脚本安装环境和下载模型。同时，建议使用微调之后的GptSovits模型进行部署。

模型转换过程如下：

1. 拷贝当前目录下文件，包含GPT_SoVITS目录和export_onnx_v2.py到GPT-Sovits项目目录（当前验证过的GPT-Sovits commit为`6df61f58e4d18d4c2ad9d1eddd6a1bd690034c23`），对原来的同级别目录进行覆盖。这个操作会修改原有模型的结构，减少大量内存拷贝并提速
2. 在GPT-SoVITS的主目录下执行脚本，导出所有需要的onnx模型，例如：`python GPT_SoVITS/export_onnx_v2.py --model_path /home/qiang/models/gpt-sovits-simple/custom --export_name custom`
3. 拷贝./GPT_SoVITS/text/G2PWModel/g2pW.onnx 到onnx模型目录下
4. 执行optimize_aio.py，对导出的onnx模型进行后期优化，默认开启int8量化，加快运行速度并减少内存占用，如果您对质量有较高要求，可以使用`--no-quant` 参数关闭量化
