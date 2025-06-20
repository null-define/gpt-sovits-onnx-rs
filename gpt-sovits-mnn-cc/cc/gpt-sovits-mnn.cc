// src/tts_mnn.cpp
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <chrono>
#include <thread>
#include "gpt-sovits-mnn.h"

extern "C"
{

    struct MNNModel
    {
        std::unique_ptr<MNN::Interpreter> sovits;
        std::unique_ptr<MNN::Interpreter> ssl;
        std::unique_ptr<MNN::Interpreter> t2s_encoder;
        std::unique_ptr<MNN::Interpreter> t2s_fs_decoder;
        std::unique_ptr<MNN::Interpreter> t2s_s_decoder;
        MNN::Session *sovits_session;
        MNN::Session *ssl_session;
        MNN::Session *t2s_encoder_session;
        MNN::Session *t2s_fs_decoder_session;
        MNN::Session *t2s_s_decoder_session;
        size_t num_layers;
    };

    MNNModel *create_mnn_model(
        const char *sovits_path,
        const char *ssl_path,
        const char *t2s_encoder_path,
        const char *t2s_fs_decoder_path,
        const char *t2s_s_decoder_path,
        size_t num_layers)
    {
        try
        {
            MNNModel *model = new MNNModel();
            model->num_layers = num_layers;
            model->t2s_encoder = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(t2s_encoder_path));
            model->t2s_fs_decoder = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(t2s_fs_decoder_path));
            model->t2s_s_decoder = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(t2s_s_decoder_path));

            model->sovits = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(sovits_path));
            model->ssl = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(ssl_path));

            if (!model->sovits || !model->ssl || !model->t2s_encoder || !model->t2s_fs_decoder || !model->t2s_s_decoder)
            {
                delete model;
                return nullptr;
            }

            MNN::ScheduleConfig config;
            config.numThread = 8;
            config.type = MNN_FORWARD_CPU;

            model->sovits_session = model->sovits->createSession(config);
            model->ssl_session = model->ssl->createSession(config);
            model->t2s_encoder_session = model->t2s_encoder->createSession(config);
            model->t2s_fs_decoder_session = model->t2s_fs_decoder->createSession(config);
            model->t2s_s_decoder_session = model->t2s_s_decoder->createSession(config);

            if (!model->sovits_session || !model->ssl_session || !model->t2s_encoder_session ||
                !model->t2s_fs_decoder_session || !model->t2s_s_decoder_session)
            {
                delete model;
                return nullptr;
            }

            return model;
        }
        catch (...)
        {
            return nullptr;
        }
    }

    void destroy_mnn_model(MNNModel *model)
    {
        if (model)
        {
            if (model->sovits && model->sovits_session)
                model->sovits->releaseSession(model->sovits_session);
            if (model->ssl && model->ssl_session)
                model->ssl->releaseSession(model->ssl_session);
            if (model->t2s_encoder && model->t2s_encoder_session)
                model->t2s_encoder->releaseSession(model->t2s_encoder_session);
            if (model->t2s_fs_decoder && model->t2s_fs_decoder_session)
                model->t2s_fs_decoder->releaseSession(model->t2s_fs_decoder_session);
            if (model->t2s_s_decoder && model->t2s_s_decoder_session)
                model->t2s_s_decoder->releaseSession(model->t2s_s_decoder_session);
            delete model;
        }
    }

    int run_ssl(
        MNNModel *model,
        const float *ref_audio_data,
        size_t ref_audio_size,
        float **ssl_output,
        size_t *ssl_output_size)
    {
        try
        {
            if (!model || !ref_audio_data || !ssl_output || !ssl_output_size)
                return -1;

            auto input_tensor = model->ssl->getSessionInput(model->ssl_session, "ref_audio_16k");
            auto output_tensor = model->ssl->getSessionOutput(model->ssl_session, "ssl_content");

            if (!input_tensor || !output_tensor)
                return -1;

            std::vector<int> input_dims = {1, static_cast<int>(ref_audio_size)};
            model->ssl->resizeTensor(input_tensor, input_dims);
            model->ssl->resizeSession(model->ssl_session);

            memcpy(input_tensor->host<float>(), ref_audio_data, ref_audio_size * sizeof(float));

            model->ssl->runSession(model->ssl_session);

            auto output_dims = output_tensor->shape();
            size_t output_size = 1;
            for (int dim : output_dims)
                output_size *= dim;

            *ssl_output = new float[output_size];
            memcpy(*ssl_output, output_tensor->host<float>(), output_size * sizeof(float));
            *ssl_output_size = output_size;

            return 0;
        }
        catch (...)
        {
            return -1;
        }
    }

    int run_inference(
        MNNModel *model,
        const int64_t *ref_seq_data,
        size_t ref_seq_size,
        const float *ref_bert_data,
        size_t ref_bert_size,
        const int64_t *text_seq_data,
        size_t text_seq_size,
        const float *text_bert_data,
        size_t text_bert_size,
        const float *ssl_content_data,
        size_t ssl_content_size,
        const float *ref_audio_data,
        size_t ref_audio_size,
        float **output_data,
        size_t *output_size)
    {
        try
        {
            if (!model || !ref_seq_data || !ref_bert_data || !text_seq_data || !text_bert_data ||
                !ssl_content_data || !ref_audio_data || !output_data || !output_size)
                return -1;

            // T2S Encoder
            auto input_ref_seq = model->t2s_encoder->getSessionInput(model->t2s_encoder_session, "ref_seq");
            auto input_ref_bert = model->t2s_encoder->getSessionInput(model->t2s_encoder_session, "ref_bert");
            auto input_text_seq = model->t2s_encoder->getSessionInput(model->t2s_encoder_session, "text_seq");
            auto input_text_bert = model->t2s_encoder->getSessionInput(model->t2s_encoder_session, "text_bert");
            auto input_ssl_content = model->t2s_encoder->getSessionInput(model->t2s_encoder_session, "ssl_content");
            auto output_x = model->t2s_encoder->getSessionOutput(model->t2s_encoder_session, "x");
            auto output_prompts = model->t2s_encoder->getSessionOutput(model->t2s_encoder_session, "prompts");

            if (!input_ref_seq || !input_ref_bert || !input_text_seq || !input_text_bert ||
                !input_ssl_content || !output_x || !output_prompts)
                return -1;
            std::cout << "set inputs" << std::endl;
            model->t2s_encoder->resizeTensor(input_ref_seq, {1, static_cast<int>(ref_seq_size)});
            model->t2s_encoder->resizeTensor(input_ref_bert, {static_cast<int>(ref_bert_size / 1024), 1024});
            model->t2s_encoder->resizeTensor(input_text_seq, {1, static_cast<int>(text_seq_size)});
            model->t2s_encoder->resizeTensor(input_text_bert, {static_cast<int>(text_bert_size / 1024), 1024});
            model->t2s_encoder->resizeTensor(input_ssl_content, {1, 768, static_cast<int>(ssl_content_size / 768)});
            model->t2s_encoder->resizeSession(model->t2s_encoder_session);
            std::cout << "set inputs 1" << std::endl;

            memcpy(input_ref_seq->host<int64_t>(), ref_seq_data, ref_seq_size * sizeof(int64_t));
            memcpy(input_ref_bert->host<float>(), ref_bert_data, ref_bert_size * sizeof(float));
            memcpy(input_text_seq->host<int64_t>(), text_seq_data, text_seq_size * sizeof(int64_t));
            memcpy(input_text_bert->host<float>(), text_bert_data, text_bert_size * sizeof(float));
            memcpy(input_ssl_content->host<float>(), ssl_content_data, ssl_content_size * sizeof(float));
            std::cout << "set inputs 2" << std::endl;

            model->t2s_encoder->runSession(model->t2s_encoder_session);
            std::cout << "run t2s" << std::endl;

            auto x_dims = output_x->shape();
            size_t x_size = 1;
            for (int dim : x_dims)
                x_size *= dim;
            std::vector<float> x_data(x_size);
            memcpy(x_data.data(), output_x->host<float>(), x_size * sizeof(float));

            auto prompts_dims = output_prompts->shape();
            size_t prompts_size = 1;
            for (int dim : prompts_dims)
                prompts_size *= dim;
            std::vector<int64_t> prompts_data(prompts_size);
            memcpy(prompts_data.data(), output_prompts->host<int64_t>(), prompts_size * sizeof(int64_t));
            auto out = model->t2s_fs_decoder->getSessionOutputAll(model->t2s_fs_decoder_session);
            // T2S FS Decoder
            auto input_x = model->t2s_fs_decoder->getSessionInput(model->t2s_fs_decoder_session, "x");
            auto input_prompts = model->t2s_fs_decoder->getSessionInput(model->t2s_fs_decoder_session, "prompts");
            auto output_y = model->t2s_fs_decoder->getSessionOutput(model->t2s_fs_decoder_session, "y");
            auto output_y_emb = model->t2s_fs_decoder->getSessionOutput(model->t2s_fs_decoder_session, "y_emb");
            auto output_x_example = model->t2s_fs_decoder->getSessionOutput(model->t2s_fs_decoder_session, "x_example");

            if (!input_x || !input_prompts || !output_y || !output_y_emb || !output_x_example)
                return -1;

            model->t2s_fs_decoder->resizeTensor(input_x, {1, static_cast<int>(x_size / 512), 512});
            model->t2s_fs_decoder->resizeTensor(input_prompts, {1, static_cast<int>(prompts_size)});
            model->t2s_fs_decoder->resizeSession(model->t2s_fs_decoder_session);
            std::cout << "run t2s fs decoder" << std::endl;

            memcpy(input_x->host<float>(), x_data.data(), x_size * sizeof(float));
            memcpy(input_prompts->host<int64_t>(), prompts_data.data(), prompts_size * sizeof(int64_t));

            model->t2s_fs_decoder->runSession(model->t2s_fs_decoder_session);
            std::cout << "run t2s fs decoder" << std::endl;

            auto y_dims = output_y->shape();
            size_t y_size = 1;
            for (int dim : y_dims)
                y_size *= dim;
            std::vector<int64_t> y_data(y_size);
            memcpy(y_data.data(), output_y->host<int64_t>(), y_size * sizeof(int64_t));

            auto y_emb_dims = output_y_emb->shape();
            size_t y_emb_size = 1;
            for (int dim : y_emb_dims)
                y_emb_size *= dim;
            std::vector<float> y_emb_data(y_emb_size);
            memcpy(y_emb_data.data(), output_y_emb->host<float>(), y_emb_size * sizeof(float));

            auto x_example_dims = output_x_example->shape();
            size_t x_example_size = 1;
            for (int dim : x_example_dims)
                x_example_size *= dim;
            std::vector<float> x_example_data(x_example_size);
            memcpy(x_example_data.data(), output_x_example->host<float>(), x_example_size * sizeof(float));

            std::vector<float *> k_caches(model->num_layers);
            std::vector<float *> v_caches(model->num_layers);
            std::vector<size_t> k_cache_sizes(model->num_layers);
            std::vector<size_t> v_cache_sizes(model->num_layers);

            for (size_t i = 0; i < model->num_layers; ++i)
            {

                std::string k_cache_name = "k_cache_" + std::to_string(i);
                std::string v_cache_name = "v_cache_" + std::to_string(i);
                auto k_cache = model->t2s_fs_decoder->getSessionOutput(model->t2s_fs_decoder_session, k_cache_name.c_str());
                auto v_cache = model->t2s_fs_decoder->getSessionOutput(model->t2s_fs_decoder_session, v_cache_name.c_str());

                if (!k_cache || !v_cache)
                    return -1;

                auto k_cache_dims = k_cache->shape();
                size_t k_cache_size = 1;
                for (int dim : k_cache_dims)
                    k_cache_size *= dim;
                k_caches[i] = new float[k_cache_size];
                memcpy(k_caches[i], k_cache->host<float>(), k_cache_size * sizeof(float));
                k_cache_sizes[i] = k_cache_size;

                auto v_cache_dims = v_cache->shape();
                size_t v_cache_size = 1;
                for (int dim : v_cache_dims)
                    v_cache_size *= dim;
                v_caches[i] = new float[v_cache_size];
                memcpy(v_caches[i], v_cache->host<float>(), v_cache_size * sizeof(float));
                v_cache_sizes[i] = v_cache_size;
            }

            // T2S S Decoder Loop
            auto input_iy = model->t2s_s_decoder->getSessionInput(model->t2s_s_decoder_session, "iy");
            auto input_iy_emb = model->t2s_s_decoder->getSessionInput(model->t2s_s_decoder_session, "iy_emb");
            auto input_x_example = model->t2s_s_decoder->getSessionInput(model->t2s_s_decoder_session, "x_example");
            output_y = model->t2s_s_decoder->getSessionOutput(model->t2s_s_decoder_session, "y");
            output_y_emb = model->t2s_s_decoder->getSessionOutput(model->t2s_s_decoder_session, "y_emb");

            if (!input_iy || !input_iy_emb || !input_x_example || !output_y || !output_y_emb)
                return -1;

            size_t idx = 1;
            size_t prefix_len = prompts_size;

            std::cout << "run single" << std::endl;
            std::cout << "iy_size:" << y_size << std::endl;
            std::cout << "iy_emb_size:" << y_emb_size << std::endl;
            std::cout << "x_example_size:" << x_example_size << std::endl;
            std::cout << "k_cache_sizes:" << k_cache_sizes[0] << std::endl;
            std::cout << "v_cache_sizes:" << v_cache_sizes[0] << std::endl;

            model->t2s_s_decoder->resizeTensor(input_iy, {1, static_cast<int>(y_size)});
            model->t2s_s_decoder->resizeTensor(input_iy_emb, {1, static_cast<int>(y_emb_size / 512), 512});
            model->t2s_s_decoder->resizeTensor(input_x_example, {1, static_cast<int>(x_example_size)});

            while (true)
            {
                

                for (size_t i = 0; i < model->num_layers; ++i)
                {
                    std::string ik_cache_name = "ik_cache_" + std::to_string(i);
                    std::string iv_cache_name = "iv_cache_" + std::to_string(i);
                    auto input_k_cache = model->t2s_s_decoder->getSessionInput(model->t2s_s_decoder_session, ik_cache_name.c_str());
                    auto input_v_cache = model->t2s_s_decoder->getSessionInput(model->t2s_s_decoder_session, iv_cache_name.c_str());
                    model->t2s_s_decoder->resizeTensor(input_k_cache, {static_cast<int>(k_cache_sizes[i] / (512)), 1, 512});
                    model->t2s_s_decoder->resizeTensor(input_v_cache, {static_cast<int>(v_cache_sizes[i] / (512)), 1, 512});
                }
                input_x_example = MNN::Tensor::create<float>(
                    {1, static_cast<int>(x_example_size)}, // Dimensions
                    x_example_data.data(),                 // Data pointer
                    MNN::Tensor::CAFFE                     // Dimension type
                );
                model->t2s_s_decoder->resizeSession(model->t2s_s_decoder_session);
                memcpy(input_iy->host<int64_t>(), y_data.data(), y_size * sizeof(int64_t));
                memcpy(input_iy_emb->host<float>(), y_emb_data.data(), y_emb_size * sizeof(float));
                memcpy(input_x_example->host<float>(), x_example_data.data(), x_example_size * sizeof(float));
                for (size_t i = 0; i < model->num_layers; ++i)
                {
                    std::string ik_cache_name = "ik_cache_" + std::to_string(i);
                    std::string iv_cache_name = "iv_cache_" + std::to_string(i);
                    auto input_k_cache = model->t2s_s_decoder->getSessionInput(model->t2s_s_decoder_session, ik_cache_name.c_str());
                    auto input_v_cache = model->t2s_s_decoder->getSessionInput(model->t2s_s_decoder_session, iv_cache_name.c_str());
                    memcpy(input_k_cache->host<float>(), k_caches[i], k_cache_sizes[i] * sizeof(float));
                    memcpy(input_v_cache->host<float>(), v_caches[i], v_cache_sizes[i] * sizeof(float));
                }

                model->t2s_s_decoder->runSession(model->t2s_s_decoder_session);
                std::cout << "run single done" << std::endl;

                auto y_new_dims = output_y->shape();
                size_t y_new_size = 1;
                for (int dim : y_new_dims)
                    y_new_size *= dim;
                y_data.resize(y_new_size);
                memcpy(y_data.data(), output_y->host<int64_t>(), y_new_size * sizeof(int64_t));
                y_size = y_new_size;

                auto y_emb_new_dims = output_y_emb->shape();
                size_t y_emb_new_size = 1;
                for (int dim : y_emb_new_dims)
                    y_emb_new_size *= dim;
                y_emb_data.resize(y_emb_new_size);
                memcpy(y_emb_data.data(), output_y_emb->host<float>(), y_emb_new_size * sizeof(float));
                y_emb_size = y_emb_new_size;

                for (size_t i = 0; i < model->num_layers; ++i)
                {
                    std::string k_cache_name = "k_cache_" + std::to_string(i);
                    std::string v_cache_name = "v_cache_" + std::to_string(i);
                    auto k_cache = model->t2s_s_decoder->getSessionOutput(model->t2s_s_decoder_session, k_cache_name.c_str());
                    auto v_cache = model->t2s_s_decoder->getSessionOutput(model->t2s_s_decoder_session, v_cache_name.c_str());

                    auto k_cache_dims = k_cache->shape();
                    size_t k_cache_new_size = 1;
                    for (int dim : k_cache_dims)
                        k_cache_new_size *= dim;
                    delete[] k_caches[i];
                    k_caches[i] = new float[k_cache_new_size];
                    memcpy(k_caches[i], k_cache->host<float>(), k_cache_new_size * sizeof(float));
                    k_cache_sizes[i] = k_cache_new_size;

                    auto v_cache_dims = v_cache->shape();
                    size_t v_cache_new_size = 1;
                    for (int dim : v_cache_dims)
                        v_cache_new_size *= dim;
                    delete[] v_caches[i];
                    v_caches[i] = new float[v_cache_new_size];
                    memcpy(v_caches[i], v_cache->host<float>(), v_cache_new_size * sizeof(float));
                    v_cache_sizes[i] = v_cache_new_size;
                }

                std::cout << " y_data[y_size - 1]:" << y_data[y_size - 1] << std::endl;

                if (idx > 10 && (idx >= 1500 ||
                                 (y_size > prefix_len && (y_size - prefix_len) > 1500) ||
                                 y_data[y_size - 1] == 1024))
                {
                    std::vector<int64_t> pred_semantic;
                    size_t output_len = y_size > prefix_len + 1 ? y_size - prefix_len - 1 : 0;
                    pred_semantic.resize(output_len);
                    for (size_t i = 0; i < output_len; ++i)
                    {
                        pred_semantic[i] = (y_data[prefix_len + 1 + i] == 1024) ? 0 : y_data[prefix_len + 1 + i];
                    }

                    // SoVITS
                    auto input_text_seq = model->sovits->getSessionInput(model->sovits_session, "text_seq");
                    auto input_pred_semantic = model->sovits->getSessionInput(model->sovits_session, "pred_semantic");
                    auto input_ref_audio = model->sovits->getSessionInput(model->sovits_session, "ref_audio");
                    auto output_audio = model->sovits->getSessionOutput(model->sovits_session, "audio");

                    if (!input_text_seq || !input_pred_semantic || !input_ref_audio || !output_audio)
                        return -1;

                    model->sovits->resizeTensor(input_text_seq, {1, static_cast<int>(text_seq_size)});
                    model->sovits->resizeTensor(input_pred_semantic, {1, static_cast<int>(pred_semantic.size())});
                    model->sovits->resizeTensor(input_ref_audio, {1, static_cast<int>(ref_audio_size)});
                    model->sovits->resizeSession(model->sovits_session);

                    memcpy(input_text_seq->host<int64_t>(), text_seq_data, text_seq_size * sizeof(int64_t));
                    memcpy(input_pred_semantic->host<int64_t>(), pred_semantic.data(), pred_semantic.size() * sizeof(int64_t));
                    memcpy(input_ref_audio->host<float>(), ref_audio_data, ref_audio_size * sizeof(float));

                    model->sovits->runSession(model->sovits_session);

                    auto output_dims = output_audio->shape();
                    size_t output_size_val = 1;
                    for (int dim : output_dims)
                        output_size_val *= dim;
                    *output_data = new float[output_size_val];
                    memcpy(*output_data, output_audio->host<float>(), output_size_val * sizeof(float));
                    *output_size = output_size_val;

                    for (size_t i = 0; i < model->num_layers; ++i)
                    {
                        delete[] k_caches[i];
                        delete[] v_caches[i];
                    }

                    return 0;
                }

                idx++;
            }
        }
        catch (...)
        {
            return -1;
        }
    }

} // extern "C"