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

| Platform   | Optimization     | T2S S Decoder Time | SoVITS Time | Index | E2E Time|
| ---------- | ---------------- | ------------------ | ----------- | ----- | ------- |
| **PC**     | Original FP32    | 26.489s            | 467.091ms   | 115   |  N/A    |
|            | + onnxsim        | 12.919s            | 446.199ms   | 111   |  N/A    |
|            | + optimize       | 9.601s             | 322.142ms   | 92    |  N/A    |
|            | + optimize\_aio  | 4.903s             | 281.727ms   | 81    |  N/A    |
|            | + 新模型转换脚本  | 1.32s              | 340.212ms   | 84    |  N/A    |
|            | + 20250613       | 1.11s              | 324.872ms   | 81    |  N/A    |
|            | + 20250614       | 1.05s              | 324.872ms   | 89    |  N/A    |
|            | + 20250621(int8) | 407ms              | 367.430ms   | 93    |  843ms  |
|            | - 20250712(int8) | 416.27ms           | 381.063ms   | 91    |  891ms  |
| **Mobile** | fp16 baseline    | 4.091s             | 1.018s      | 83    |  N/A    |
|            | + NNAPI (slower) | 5.337s             | 1.024s      | 77    |  N/A    |
|            | + 20250621(int8) | 775.3ms            | 1.07s       | 100   | 1977 ms |

---
