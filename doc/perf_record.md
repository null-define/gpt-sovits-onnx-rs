# Perf Record

* **Input**: 你好呀，我们是一群追逐梦想的人！
* **Output WAV Time**: 3s

## **PC**

* **Hardware**: AMD Ryzen 7 7700
* **OS**: Ubuntu 22.04.5 LTS
* **Environment**: WSL2

## **Mobile**

* **Hardware**: Qualcomm Snapdragon 8 Elite
* **OS**: Android 15

---

### **Performance Comparison Table**

| Platform   | Optimization     | T2S S Decoder Time | SoVITS Time | Index |
| ---------- | ---------------- | ------------------ | ----------- | ----- |
| **PC**     | Original FP32    | 26.489s            | 467.091ms   | 115   |
|            | + onnxsim        | 12.919s            | 446.199ms   | 111   |
|            | + optimize       | 9.601s             | 322.142ms   | 92    |
|            | optimize\_aio.py | 4.903s             | 281.727ms   | 81    |
|            | + 新模型转换脚本  | 1.32s              | 340.212ms   | 84    |
|            | + 转换脚本0613    | 1.11s              | 324.872ms   | 81    |
| **Mobile** | fp16 baseline    | 4.091s             | 1.018s      | 83    |
|            | + NNAPI (slower) | 5.337s             | 1.024s      | 77    |

---
