
extern "C"
{

    struct MNNModel;
    MNNModel *create_mnn_model(
        const char *sovits_path,
        const char *ssl_path,
        const char *t2s_encoder_path,
        const char *t2s_fs_decoder_path,
        const char *t2s_s_decoder_path,
        unsigned long num_layers);

    void destroy_mnn_model(MNNModel *model);

    int run_ssl(
        MNNModel *model,
        const float *ref_audio_data,
        unsigned long ref_audio_size,
        float **ssl_output,
        unsigned long *ssl_output_size);

    int run_inference(
        MNNModel *model,
        const long *ref_seq_data,
        unsigned long ref_seq_size,
        const float *ref_bert_data,
        unsigned long ref_bert_size,
        const long *text_seq_data,
        unsigned long text_seq_size,
        const float *text_bert_data,
        unsigned long text_bert_size,
        const float *ssl_content_data,
        unsigned long ssl_content_size,
        const float *ref_audio_data,
        unsigned long ref_audio_size,
        float **output_data,
        unsigned long *output_size);
}
