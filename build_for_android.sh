

export ORT_LIB_LOCATION=$PWD/onnx/onnxruntime-1.22.0/build/Android/Release
export ORT_INCLUDE_DIR==$PWD/onnx/onnxruntime-1.22.0/include

export TARGET=aarch64-linux-android
export API=32

# Add the NDK toolchain to your PATH
export PATH=$NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin:$PATH

# Build with cargo
cargo build --target $TARGET --release --verbose