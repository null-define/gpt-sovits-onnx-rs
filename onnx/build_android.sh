cd onnxruntime-1.22.0
./build.sh --android \
    --android_ndk_path $NDK_HOME \
    --android_sdk_path $SDK_HOME \
    --config Release \
    --android_api 32 \
    --android_abi arm64-v8a \
    --use_nnapi \
    --build_shared_lib \
    --skip_tests \
    --parallel 