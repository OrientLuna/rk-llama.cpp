#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

struct rknn_encoder {
    uint64_t ctx = 0;
    void * input_attrs  = nullptr;
    void * output_attrs = nullptr;

    int n_input  = 0;
    int n_output = 0;

    int model_channel = 3;
    int model_width   = 224;
    int model_height  = 224;
    int model_image_token = 0;
    int model_embed_size  = 0;

    int n_merge      = 3;
    int n_tokens_out = 0;

    std::vector<float> output_raw;
    std::vector<float> output_pooled;
};

rknn_encoder * rknn_encoder_init(const char * rknn_path, int n_merge, int n_embd_proj);
int rknn_encoder_run(rknn_encoder * enc, const float * pixels, int img_w, int img_h, float * output);
void rknn_encoder_destroy(rknn_encoder * enc);
