if [ ! -f "./v1.22.0.tar.gz" ]; then
    wget https://github.com/microsoft/onnxruntime/archive/refs/tags/v1.22.0.tar.gz
fi

if [ ! -d "./onnxruntime-1.22.0" ]; then
    tar -xzf v1.22.0.tar.gz
fi