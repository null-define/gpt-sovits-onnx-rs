#include "cc/gpt-sovits-mnn.h"

int main(int argc, char const *argv[])
{
    create_mnn_model(
        "/home/qiang/projects/GPT-SoVITS/mnn/kaoyu_vits.mnn",
        "/home/qiang/projects/GPT-SoVITS/mnn/kaoyu_ssl.mnn",
        "/home/qiang/projects/GPT-SoVITS/mnn/kaoyu_t2s_encoder.mnn",
        "/home/qiang/projects/GPT-SoVITS/mnn/kaoyu_t2s_fs_decoder.mnn",
        "/home/qiang/projects/GPT-SoVITS/mnn/kaoyu_t2s_s_decoder.mnn",
        24);
    return 0;
}
