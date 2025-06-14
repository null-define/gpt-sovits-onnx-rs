
#!/bin/bash

# Check if ANDROID_NDK_HOME is set
if [ -z "$ANDROID_NDK_HOME" ]; then
    echo "Error: ANDROID_NDK_HOME is not set"
    exit 1
fi

# Check if ANDROID_SDK_HOME is set
if [ -z "$ANDROID_SDK_HOME" ]; then
    echo "Error: ANDROID_SDK_HOME is not set"
    exit 1
fi

# Optional: Verify the paths exist (uncomment if needed)
if [ ! -d "$ANDROID_NDK_HOME" ]; then
    echo "Error: ANDROID_NDK_HOME path does not exist: $ANDROID_NDK_HOME"
    exit 1
fi

if [ ! -d "$ANDROID_SDK_HOME" ]; then
    echo "Error: ANDROID_SDK_HOME path does not exist: $ANDROID_SDK_HOME"
    exit 1
fi

echo "ANDROID_NDK_HOME and ANDROID_SDK_HOME are set"

cd onnxruntime-1.22.0
./build.sh --android \
    --android_ndk_path $ANDROID_NDK_HOME \
    --android_sdk_path $ANDROID_SDK_HOME \
    --config MinSizeRel \
    --android_api 32 \
    --android_abi arm64-v8a \
    --build_shared_lib \
    --skip_tests \
    --parallel \
    --use_xnnpack \
    --disable_ml_ops \
    --cmake_extra_defines onnxruntime_ENABLE_CPU_FP16_OPS=ON 
