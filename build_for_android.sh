
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

cd onnx
bash download_onnx.sh
bash build_android.sh

cd ..

export ORT_LIB_LOCATION=$PWD/onnx/onnxruntime-1.22.0/build/Android/Release
export ORT_INCLUDE_DIR==$PWD/onnx/onnxruntime-1.22.0/include

export TARGET=aarch64-linux-android
export API=32

# Add the NDK toolchain to your PATH
export PATH=$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin:$PATH

# Build with cargo
cargo build --target $TARGET --release --features jni