# 使用说明

1. 拷贝当前目录下文件，包含models/modules/export_onnx_v2.py到GPT-Sovits python项目目录，和原来的onnx export脚本同级并进行覆盖(非常重要！！！)。这个操作会修改原有模型的结构，减少大量内存拷贝并提速！
2. 执行脚本，导出所有需要的onnx模型
3. 执行optimize_aio.py