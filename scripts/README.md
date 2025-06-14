# 使用说明

1. 拷贝当前目录下文件，包含GPT_SoVITS目录和export_onnx_v2.py到GPT-Sovits项目目录，对原来的同级别目录进行覆盖。这个操作会修改原有模型的结构，减少大量内存拷贝并提速
2. 执行脚本，导出所有需要的onnx模型
3. 拷贝./GPT_SoVITS/text/G2PWModel/g2pW.onnx 到onnx模型目录下
4. 执行optimize_aio.py，对导出的onnx模型进行后期优化
