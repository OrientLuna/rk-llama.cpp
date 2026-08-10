#include "rknn_encoder.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <cinttypes>

#include "rknn_api.h"

static void dump_tensor_attr(rknn_tensor_attr * attr) {
    fprintf(stderr, "  index=%d, name=%s, n_dims=%d, dims=[%d,%d,%d,%d], n_elems=%d, size=%d, "
                    "fmt=%s, type=%s, qnt_type=%s, zp=%d, scale=%f\n",
            attr->index, attr->name, attr->n_dims,
            attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
            attr->n_elems, attr->size,
            get_format_string(attr->fmt), get_type_string(attr->type),
            get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

rknn_encoder * rknn_encoder_init(const char * rknn_path, int n_merge, int n_embd_proj) {
    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, (void *)rknn_path, 0, 0, nullptr);
    if (ret < 0) {
        fprintf(stderr, "[rknn] rknn_init fail! ret=%d path=%s\n", ret, rknn_path);
        return nullptr;
    }

    ret = rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1_2);
    if (ret < 0) {
        fprintf(stderr, "[rknn] rknn_set_core_mask fail! ret=%d\n", ret);
        rknn_destroy(ctx);
        return nullptr;
    }

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "[rknn] query IN_OUT_NUM fail! ret=%d\n", ret);
        rknn_destroy(ctx);
        return nullptr;
    }
    fprintf(stderr, "[rknn] model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    rknn_tensor_attr * input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memset(input_attrs, 0, io_num.n_input * sizeof(rknn_tensor_attr));
    fprintf(stderr, "[rknn] input tensors:\n");
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            fprintf(stderr, "[rknn] query INPUT_ATTR fail! ret=%d\n", ret);
            free(input_attrs);
            rknn_destroy(ctx);
            return nullptr;
        }
        dump_tensor_attr(&input_attrs[i]);
    }

    rknn_tensor_attr * output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memset(output_attrs, 0, io_num.n_output * sizeof(rknn_tensor_attr));
    fprintf(stderr, "[rknn] output tensors:\n");
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            fprintf(stderr, "[rknn] query OUTPUT_ATTR fail! ret=%d\n", ret);
            free(input_attrs);
            free(output_attrs);
            rknn_destroy(ctx);
            return nullptr;
        }
        dump_tensor_attr(&output_attrs[i]);
    }

    int model_image_token = 0;
    int model_embed_size  = 0;
    for (int i = 0; i < 4; i++) {
        if (output_attrs[0].dims[i] > 1) {
            model_image_token = output_attrs[0].dims[i];
            model_embed_size  = output_attrs[0].dims[i + 1];
            break;
        }
    }

    int model_channel = 3, model_width = 224, model_height = 224;
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        model_channel = input_attrs[0].dims[1];
        model_height  = input_attrs[0].dims[2];
        model_width   = input_attrs[0].dims[3];
    } else {
        model_height  = input_attrs[0].dims[1];
        model_width   = input_attrs[0].dims[2];
        model_channel = input_attrs[0].dims[3];
    }
    fprintf(stderr, "[rknn] model input: height=%d, width=%d, channel=%d\n",
            model_height, model_width, model_channel);
    fprintf(stderr, "[rknn] model output: image_token=%d, embed_size=%d\n",
            model_image_token, model_embed_size);

    int n_patches = model_image_token;
    int n_embd_out = model_embed_size;
    int side = (int)std::sqrt((float)n_patches);
    int out_side = side / n_merge;
    int n_tokens_out = out_side * out_side;

    auto * enc = new rknn_encoder();
    enc->ctx = (uint64_t)ctx;
    enc->n_input = io_num.n_input;
    enc->n_output = io_num.n_output;
    enc->input_attrs = input_attrs;
    enc->output_attrs = output_attrs;
    enc->model_channel = model_channel;
    enc->model_width = model_width;
    enc->model_height = model_height;
    enc->model_image_token = model_image_token;
    enc->model_embed_size = model_embed_size;
    enc->n_merge = n_merge;
    enc->n_tokens_out = n_tokens_out;
    enc->output_raw.resize(n_patches * n_embd_out);
    enc->output_pooled.resize(n_tokens_out * n_embd_out);

    fprintf(stderr, "[rknn] OK: %d patches x %d embd -> pooled %d tokens (n_merge=%d)\n",
            n_patches, n_embd_out, n_tokens_out, n_merge);

    return enc;
}

int rknn_encoder_run(rknn_encoder * enc, const float * pixels, int img_w, int img_h, float * output) {
    if (!enc || !enc->ctx) return -1;
    rknn_context ctx = (rknn_context)enc->ctx;

    const int target_w = enc->model_width;
    const int target_h = enc->model_height;
    const int ch = enc->model_channel;
    std::vector<uint8_t> input_uint8(target_h * target_w * ch);

    float sx = (float)img_w / target_w;
    float sy = (float)img_h / target_h;
    for (int y = 0; y < target_h; y++) {
        float src_y = (y + 0.5f) * sy - 0.5f;
        if (src_y < 0) src_y = 0;
        if (src_y > img_h - 1) src_y = img_h - 1;
        int iy0 = (int)src_y;
        int iy1 = (iy0 + 1 < img_h) ? iy0 + 1 : iy0;
        float fy = src_y - iy0;
        for (int x = 0; x < target_w; x++) {
            float src_x = (x + 0.5f) * sx - 0.5f;
            if (src_x < 0) src_x = 0;
            if (src_x > img_w - 1) src_x = img_w - 1;
            int ix0 = (int)src_x;
            int ix1 = (ix0 + 1 < img_w) ? ix0 + 1 : ix0;
            float fx = src_x - ix0;
            for (int c = 0; c < ch; c++) {
                float v00 = pixels[c * img_h * img_w + iy0 * img_w + ix0];
                float v10 = pixels[c * img_h * img_w + iy0 * img_w + ix1];
                float v01 = pixels[c * img_h * img_w + iy1 * img_w + ix0];
                float v11 = pixels[c * img_h * img_w + iy1 * img_w + ix1];
                float v = ((v00 * (1 - fx) + v10 * fx) * (1 - fy) +
                           (v01 * (1 - fx) + v11 * fx) * fy);
                int val = (int)(v * 255.0f + 0.5f);
                if (val < 0) val = 0;
                if (val > 255) val = 255;
                input_uint8[(y * target_w + x) * ch + c] = (uint8_t)val;
            }
        }
    }

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type  = RKNN_TENSOR_UINT8;
    inputs[0].fmt   = RKNN_TENSOR_NHWC;
    inputs[0].size  = target_w * target_h * ch;
    inputs[0].buf   = input_uint8.data();
    int ret = rknn_inputs_set(ctx, 1, inputs);
    if (ret < 0) {
        fprintf(stderr, "[rknn] inputs_set fail! ret=%d\n", ret);
        return -1;
    }

    ret = rknn_run(ctx, nullptr);
    if (ret < 0) {
        fprintf(stderr, "[rknn] run fail! ret=%d\n", ret);
        return -1;
    }

    int n_output = enc->n_output;
    std::vector<rknn_output> outputs(n_output);
    memset(outputs.data(), 0, outputs.size() * sizeof(rknn_output));
    for (int j = 0; j < n_output; j++) {
        outputs[j].want_float = 1;
    }
    ret = rknn_outputs_get(ctx, n_output, outputs.data(), nullptr);
    if (ret < 0) {
        fprintf(stderr, "[rknn] outputs_get fail! ret=%d\n", ret);
        return -1;
    }

    int n_patches = enc->model_image_token;
    int n_embd = enc->model_embed_size;

    if (n_output == 1) {
        memcpy(enc->output_raw.data(), outputs[0].buf, outputs[0].size);
    } else {
        for (int i = 0; i < n_patches; i++) {
            for (int j = 0; j < n_output; j++) {
                memcpy(enc->output_raw.data() + i * n_output * n_embd + j * n_embd,
                       (float *)outputs[j].buf + i * n_embd,
                       sizeof(float) * n_embd);
            }
        }
    }
    rknn_outputs_release(ctx, n_output, outputs.data());

    int side = (int)std::sqrt((float)n_patches);
    int out_side = side / enc->n_merge;
    int pk = enc->n_merge;
    for (int oy = 0; oy < out_side; oy++)
        for (int ox = 0; ox < out_side; ox++)
            for (int c = 0; c < n_embd; c++) {
                float sum = 0.0f;
                for (int ky = 0; ky < pk; ky++)
                    for (int kx = 0; kx < pk; kx++)
                        sum += enc->output_raw[((oy * pk + ky) * side + (ox * pk + kx)) * n_embd + c];
                enc->output_pooled[(oy * out_side + ox) * n_embd + c] = sum / (pk * pk);
            }

    memcpy(output, enc->output_pooled.data(), enc->n_tokens_out * n_embd * sizeof(float));
    return enc->n_tokens_out;
}

void rknn_encoder_destroy(rknn_encoder * enc) {
    if (!enc) return;
    if (enc->input_attrs) { free(enc->input_attrs); enc->input_attrs = nullptr; }
    if (enc->output_attrs) { free(enc->output_attrs); enc->output_attrs = nullptr; }
    if (enc->ctx) rknn_destroy((rknn_context)enc->ctx);
    delete enc;
}
