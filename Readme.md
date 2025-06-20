# GPT-SOVITS-ONNX-RS (MNN BACKUP)

⚠️ DO NOT USE, STILL NOT AVAILABLE FOR END 2 END INFERENCE, JUST DEV CODE BACKUP!!

⚠️ 不要使用这个分支!!目前并未开发完成，后续可能也不会再继续开发。

这是MNN方案的备份分支，正如我在主分支中的readme所提到的，目前MNN仍然无法正确加载decoder模型的输入/输出，所以无法跑完整个流程。

如果您是MNN专家，或者愿意修复这个BUG，可以在当前分支的代码上进行修改。

⚠️ 注意！！不要将MNN的build过程写入build.rs，因为MNN尝试连接protobuf，但是一旦rust项目中使用的第三方项目也使用了protobuf，那么在build.rs中构建的MNN可能链接错误的版本，进而导致模型加载失败！！