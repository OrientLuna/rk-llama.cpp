#ifdef GGML_USE_RKNPU2

#include "rkllm-instance.h"

#include <cstring>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <nlohmann/json.hpp>

// stb_image for image decoding (implementation provided by mtmd-helper.cpp)
#include "stb_image.h"

extern "C" {
#include "rkllm.h"
#include "rknn_api.h"
}

using json = nlohmann::json;

#define RKLLM_LOG(...) fprintf(stderr, "[rkllm] " __VA_ARGS__)
#define RKLLM_INF(...) fprintf(stderr, "[rkllm] " __VA_ARGS__)

// ── RKNN vision encoder context (mirrors image_enc.h from multimodal demo) ──

struct rknn_app_context_t {
    rknn_context rknn_ctx = 0;
    struct { int n_input = 0; int n_output = 0; } io_num;
    rknn_tensor_attr * input_attrs  = nullptr;
    rknn_tensor_attr * output_attrs = nullptr;
    int model_channel = 3;
    int model_width   = 224;
    int model_height  = 224;
    int model_image_token = 0;
    int model_embed_size  = 0;
};

// ── RKNN vision encoder init/run (adapted from image_enc.cc) ──

static bool rknn_enc_init(const char * model_path, rknn_app_context_t * ctx) {
    int ret = rknn_init(&ctx->rknn_ctx, (void *)model_path, 0, 0, nullptr);
    if (ret < 0) { RKLLM_LOG("vision rknn_init fail ret=%d\n", ret); return false; }

    ret = rknn_set_core_mask(ctx->rknn_ctx, RKNN_NPU_CORE_0_1_2);

    rknn_input_output_num io_num;
    ret = rknn_query(ctx->rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) return false;
    ctx->io_num.n_input  = io_num.n_input;
    ctx->io_num.n_output = io_num.n_output;

    ctx->input_attrs = (rknn_tensor_attr *)calloc(io_num.n_input, sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num.n_input; i++) {
        ctx->input_attrs[i].index = i;
        rknn_query(ctx->rknn_ctx, RKNN_QUERY_INPUT_ATTR, &ctx->input_attrs[i], sizeof(rknn_tensor_attr));
    }
    ctx->output_attrs = (rknn_tensor_attr *)calloc(io_num.n_output, sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num.n_output; i++) {
        ctx->output_attrs[i].index = i;
        rknn_query(ctx->rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, &ctx->output_attrs[i], sizeof(rknn_tensor_attr));
    }

    // infer image_token and embed_size from output dims
    for (int i = 0; i < 4; i++) {
        if (ctx->output_attrs[0].dims[i] > 1) {
            ctx->model_image_token = ctx->output_attrs[0].dims[i];
            ctx->model_embed_size  = ctx->output_attrs[0].dims[i + 1];
            break;
        }
    }

    // infer input H/W/C
    if (ctx->input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        ctx->model_channel = ctx->input_attrs[0].dims[1];
        ctx->model_height  = ctx->input_attrs[0].dims[2];
        ctx->model_width   = ctx->input_attrs[0].dims[3];
    } else {
        ctx->model_height  = ctx->input_attrs[0].dims[1];
        ctx->model_width   = ctx->input_attrs[0].dims[2];
        ctx->model_channel = ctx->input_attrs[0].dims[3];
    }

    RKLLM_INF("vision encoder: %dx%dx%d, tokens=%d, embed=%d, outputs=%d\n",
        ctx->model_width, ctx->model_height, ctx->model_channel,
        ctx->model_image_token, ctx->model_embed_size, ctx->io_num.n_output);
    return true;
}

static bool rknn_enc_run(rknn_app_context_t * ctx, const uint8_t * img_rgb, std::vector<float> & out) {
    int n_out = ctx->io_num.n_output;
    int total = ctx->model_image_token * ctx->model_embed_size * n_out;
    out.resize(total);

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type  = RKNN_TENSOR_UINT8;
    inputs[0].fmt   = RKNN_TENSOR_NHWC;
    inputs[0].size  = ctx->model_width * ctx->model_height * ctx->model_channel;
    inputs[0].buf   = (void *)img_rgb;

    if (rknn_inputs_set(ctx->rknn_ctx, 1, inputs) < 0) return false;
    if (rknn_run(ctx->rknn_ctx, nullptr) < 0) return false;

    std::vector<rknn_output> outputs(n_out);
    memset(outputs.data(), 0, n_out * sizeof(rknn_output));
    for (int j = 0; j < n_out; j++) outputs[j].want_float = 1;

    if (rknn_outputs_get(ctx->rknn_ctx, n_out, outputs.data(), nullptr) < 0) return false;

    if (n_out == 1) {
        memcpy(out.data(), outputs[0].buf, outputs[0].size);
    } else {
        for (int i = 0; i < ctx->model_image_token; i++) {
            for (int j = 0; j < n_out; j++) {
                memcpy(out.data() + i * n_out * ctx->model_embed_size + j * ctx->model_embed_size,
                       (float *)outputs[j].buf + i * ctx->model_embed_size,
                       sizeof(float) * ctx->model_embed_size);
            }
        }
    }
    rknn_outputs_release(ctx->rknn_ctx, n_out, outputs.data());
    return true;
}

static void rknn_enc_destroy(rknn_app_context_t * ctx) {
    if (!ctx) return;
    if (ctx->input_attrs)  { free(ctx->input_attrs);  ctx->input_attrs = nullptr; }
    if (ctx->output_attrs) { free(ctx->output_attrs); ctx->output_attrs = nullptr; }
    if (ctx->rknn_ctx)     { rknn_destroy(ctx->rknn_ctx); ctx->rknn_ctx = 0; }
    delete ctx;
}

// ── Image resize (bilinear, no OpenCV) ──

static void resize_rgb_bilinear(const uint8_t * src, int src_w, int src_h,
                                 uint8_t * dst, int dst_w, int dst_h) {
    float sx = (float)src_w / dst_w;
    float sy = (float)src_h / dst_h;
    for (int y = 0; y < dst_h; y++) {
        float src_y = (y + 0.5f) * sy - 0.5f;
        if (src_y < 0) src_y = 0;
        if (src_y > src_h - 1) src_y = src_h - 1;
        int iy0 = (int)src_y, iy1 = (iy0 + 1 < src_h) ? iy0 + 1 : iy0;
        float fy = src_y - iy0;
        for (int x = 0; x < dst_w; x++) {
            float src_x = (x + 0.5f) * sx - 0.5f;
            if (src_x < 0) src_x = 0;
            if (src_x > src_w - 1) src_x = src_w - 1;
            int ix0 = (int)src_x, ix1 = (ix0 + 1 < src_w) ? ix0 + 1 : ix0;
            float fx = src_x - ix0;
            for (int c = 0; c < 3; c++) {
                float v00 = src[(iy0 * src_w + ix0) * 3 + c];
                float v10 = src[(iy0 * src_w + ix1) * 3 + c];
                float v01 = src[(iy1 * src_w + ix0) * 3 + c];
                float v11 = src[(iy1 * src_w + ix1) * 3 + c];
                int val = (int)(((v00 * (1-fx) + v10 * fx) * (1-fy) +
                                 (v01 * (1-fx) + v11 * fx) * fy) + 0.5f);
                dst[(y * dst_w + x) * 3 + c] = (uint8_t)std::max(0, std::min(255, val));
            }
        }
    }
}

// ── RKLLM callback → stream_ctx bridge ──

static int rkllm_result_callback(RKLLMResult * result, void * userdata, LLMCallState state) {
    auto * ctx = static_cast<rkllm_model_instance::stream_ctx *>(userdata);
    if (!ctx) return 0;
    switch (state) {
        case RKLLM_RUN_NORMAL:
            if (result) {
                // Use NORMAL callbacks as a live output estimate. The finish
                // callback replaces this estimate with the SDK's exact value.
                ctx->note_generated_token();
                if (result->text && result->text[0]) ctx->push(std::string(result->text));
            }
            break;
        case RKLLM_RUN_FINISH:
            if (result) {
                ctx->set_perf(result->perf.prefill_time_ms, result->perf.prefill_tokens,
                              result->perf.generate_time_ms, result->perf.generate_tokens);
            }
            ctx->done.store(true); ctx->cv.notify_all(); break;
        case RKLLM_RUN_ERROR:
            ctx->error.store(true); ctx->cv.notify_all(); break;
        default: break;
    }
    if (ctx->abort_requested.load()) return 1;
    return 0;
}

// ── SSE formatting ──

std::string rkllm_model_instance::sse_chunk(const std::string & model, const std::string & content, const std::string & finish_reason) {
    return sse_chunk(model, content, finish_reason, json(nullptr));
}

std::string rkllm_model_instance::sse_chunk(const std::string & model, const std::string & content,
                                            const std::string & finish_reason, const json & timings) {
    json chunk = {
        {"object", "chat.completion.chunk"}, {"model", model},
        {"choices", json::array({{
            {"delta", content.empty() && finish_reason.empty() ? json{{"role","assistant"}} : json{{"content", content}}},
            {"finish_reason", finish_reason.empty() ? json(nullptr) : json(finish_reason)},
        }})},
    };
    if (!timings.is_null()) chunk["timings"] = timings;
    return "data: " + chunk.dump() + "\n\n";
}

std::string rkllm_model_instance::sse_usage_chunk(const std::string & model, const json & usage, const json & timings) {
    json chunk = {
        {"object", "chat.completion.chunk"},
        {"model", model},
        {"choices", json::array()},
        {"usage", usage},
    };
    if (!timings.is_null()) chunk["timings"] = timings;
    return "data: " + chunk.dump() + "\n\n";
}

std::string rkllm_model_instance::sse_done() { return "data: [DONE]\n\n"; }

void rkllm_model_instance::stream_ctx::note_generated_token() {
    std::lock_guard<std::mutex> lk(metrics_mtx);
    if (!generation_started) {
        generation_started = true;
        generation_started_at = std::chrono::steady_clock::now();
    }
    ++generated_tokens;
}

void rkllm_model_instance::stream_ctx::set_perf(float prompt_ms, int prompt_tokens,
                                                  float generated_ms, int generated_tokens) {
    std::lock_guard<std::mutex> lk(metrics_mtx);
    perf.prompt_ms = prompt_ms;
    perf.prompt_tokens = prompt_tokens;
    perf.generated_ms = generated_ms;
    perf.generated_tokens = generated_tokens;
    perf.has_perf = true;
}

rkllm_model_instance::stream_ctx::metrics_snapshot rkllm_model_instance::stream_ctx::get_metrics() const {
    std::lock_guard<std::mutex> lk(metrics_mtx);
    metrics_snapshot result = perf;
    if (!result.has_perf) {
        result.generated_tokens = generated_tokens;
        if (generation_started) {
            result.generated_ms = std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - generation_started_at).count();
        }
    }
    return result;
}

static json rkllm_usage_json(const rkllm_model_instance::stream_ctx & ctx) {
    const auto metrics = ctx.get_metrics();
    return {
        {"completion_tokens", metrics.generated_tokens},
        {"prompt_tokens", metrics.prompt_tokens},
        {"total_tokens", metrics.generated_tokens + metrics.prompt_tokens},
        {"prompt_tokens_details", {{"cached_tokens", 0}}},
    };
}

static json rkllm_timings_json(const rkllm_model_instance::stream_ctx & ctx) {
    const auto metrics = ctx.get_metrics();
    const double prompt_ms = metrics.prompt_ms;
    const double generated_ms = metrics.generated_ms;
    const int prompt_tokens = metrics.prompt_tokens;
    const int generated_tokens = metrics.generated_tokens;
    return {
        {"cache_n", 0},
        {"prompt_n", prompt_tokens},
        {"prompt_ms", prompt_ms},
        {"prompt_per_token_ms", prompt_tokens > 0 ? prompt_ms / prompt_tokens : 0.0},
        {"prompt_per_second", prompt_ms > 0 ? 1000.0 * prompt_tokens / prompt_ms : 0.0},
        {"predicted_n", generated_tokens},
        {"predicted_ms", generated_ms},
        {"predicted_per_token_ms", generated_tokens > 0 ? generated_ms / generated_tokens : 0.0},
        {"predicted_per_second", generated_ms > 0 ? 1000.0 * generated_tokens / generated_ms : 0.0},
    };
}

// ── Lifecycle ──

rkllm_model_instance::~rkllm_model_instance() { unload(); }

bool rkllm_model_instance::load(const std::string & model_path, int max_context_len,
                                 const std::string & vision_encoder_path) {
    if (loaded_) return true;

    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = model_path.c_str();
    param.max_context_len = max_context_len;
    // max_new_tokens: -1 means "generate until EOS", but RKLLM v1.2.0 interprets it as ~1 token.
    // Official demo uses 2048. Use 512 as safe default (can be overridden via request).
    param.max_new_tokens  = 512;
    param.top_k = 1; param.top_p = 0.9; param.temperature = 0.8;
    param.repeat_penalty = 1.1f; param.skip_special_token = true;
    param.extend_param.base_domain_id = 0;
    param.extend_param.embed_flash   = 1;
    param.extend_param.n_batch       = 1;

#if RKLLM_API_ABI_VERSION >= 130
    // SDK 1.3.0 uses a callback structure; optional tokenizer/embed callbacks
    // remain null because the deployed models have internal implementations.
    RKLLMCallback callback{};
    callback.result_callback = rkllm_result_callback;
    int ret = rkllm_init(&handle_, &param, &callback);
#else
    // SDK 1.2.3 uses the raw result callback ABI.
    int ret = rkllm_init(&handle_, &param, rkllm_result_callback);
#endif
    if (ret != 0) { RKLLM_LOG("rkllm_init failed ret=%d model=%s\n", ret, model_path.c_str()); return false; }
    loaded_ = true;
    RKLLM_INF("LLM loaded: %s\n", model_path.c_str());

    // Load vision encoder if specified
    if (!vision_encoder_path.empty()) {
        vision_enc_ = new rknn_app_context_t();
        memset(vision_enc_, 0, sizeof(rknn_app_context_t));
        if (rknn_enc_init(vision_encoder_path.c_str(), vision_enc_)) {
            vision_width_      = vision_enc_->model_width;
            vision_height_     = vision_enc_->model_height;
            vision_image_token_ = vision_enc_->model_image_token;
            vision_embed_size_  = vision_enc_->model_embed_size;
            vision_n_output_    = vision_enc_->io_num.n_output;
            RKLLM_INF("vision encoder loaded: %s\n", vision_encoder_path.c_str());
        } else {
            RKLLM_LOG("vision encoder init failed: %s\n", vision_encoder_path.c_str());
            rknn_enc_destroy(vision_enc_);
            vision_enc_ = nullptr;
        }
    }
    return true;
}

void rkllm_model_instance::abort_inference() {
    if (handle_) rkllm_abort(handle_);
    if (worker_.joinable()) worker_.join();
}

void rkllm_model_instance::unload() {
    abort_inference();
    if (vision_enc_) { rknn_enc_destroy(vision_enc_); vision_enc_ = nullptr; }
    if (handle_) { rkllm_destroy(handle_); handle_ = nullptr; loaded_ = false; RKLLM_INF("unloaded\n"); }
}

// ── Image encoding ──

bool rkllm_model_instance::encode_image(const std::string & image_path, std::vector<float> & embed, int & n_tokens) {
    if (!vision_enc_) return false;

    // Decode image (stb_image → RGB)
    int img_w, img_h, img_ch;
    uint8_t * img_data = stbi_load(image_path.c_str(), &img_w, &img_h, &img_ch, 3);
    if (!img_data) { RKLLM_LOG("stbi_load failed: %s\n", image_path.c_str()); return false; }

    // Resize to model input size
    std::vector<uint8_t> resized(vision_width_ * vision_height_ * 3);
    resize_rgb_bilinear(img_data, img_w, img_h, resized.data(), vision_width_, vision_height_);
    stbi_image_free(img_data);

    // Run RKNN vision encoder
    bool ok = rknn_enc_run(vision_enc_, resized.data(), embed);
    n_tokens = vision_image_token_;
    return ok;
}

// ── Chat completion ──

server_http_res_ptr rkllm_model_instance::chat_completion(
    const std::string & model_name,
    const std::string & body_json,
    const std::function<bool()> & should_stop
) {
    model_name_ = model_name;
    auto res = std::make_unique<server_http_res>();
    res->status = 200;

    json body;
    try { body = json::parse(body_json); }
    catch (...) { res->content_type = "application/json"; res->data = "{\"error\":{\"message\":\"invalid JSON\"}}"; res->status = 400; return res; }

    bool stream = body.value("stream", false);

    // Extract the OpenAI names for the generation limit. RKLLM otherwise uses
    // the init-time value, which made max_tokens appear to be ignored.
    int max_tokens = body.value("max_completion_tokens", body.value("max_tokens", 512));
    if (max_tokens <= 0) max_tokens = 512;
    const bool timings_per_token = body.value("timings_per_token", false);
    bool include_usage = false;
    if (body.contains("stream_options") && body["stream_options"].is_object()) {
        include_usage = body["stream_options"].value("include_usage", false);
    }

    // Extract prompt + image from messages
    std::string prompt;
    std::string role = "user";
    std::string image_path;  // local file path if multimodal

    if (body.contains("messages") && !body["messages"].empty()) {
        auto & last_msg = body["messages"].back();
        role = last_msg.value("role", "user");
        if (last_msg.contains("content")) {
            if (last_msg["content"].is_string()) {
                prompt = last_msg["content"].get<std::string>();
            } else if (last_msg["content"].is_array()) {
                for (auto & part : last_msg["content"]) {
                    std::string type = part.value("type", "");
                    if (type == "text") {
                        prompt = part.value("text", "");
                    } else if (type == "image_url" && part.contains("image_url")) {
                        std::string url = part["image_url"].value("url", "");
                        if (url.size() > 8 && url.substr(0, 8) == "file:///") {
                            image_path = url.substr(7);
                        } else if (!url.empty() && url[0] == '/') {
                            image_path = url;
                        } else if (url.size() > 5 && url.substr(0, 5) == "data:") {
                            // base64 data URI: data:image/jpeg;base64,<data>
                            // Decode and write to temp file
                            size_t b64_start = url.find("base64,");
                            if (b64_start != std::string::npos) {
                                std::string b64_data = url.substr(b64_start + 7);
                                // base64 decode
                                static const int b64_table[] = {
                                    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
                                    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
                                    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
                                    52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,
                                    -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
                                    15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
                                    -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
                                    41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1
                                };
                                std::vector<uint8_t> decoded;
                                int val = 0, valb = -8;
                                for (unsigned char c : b64_data) {
                                    if (c >= 128 || b64_table[c] < 0) {
                                        if (c == '=') break;
                                        continue;
                                    }
                                    val = (val << 6) + b64_table[c];
                                    valb += 6;
                                    if (valb >= 0) {
                                        decoded.push_back((val >> valb) & 0xFF);
                                        valb -= 8;
                                    }
                                }
                                // Write to temp file
                                image_path = "/tmp/rkllm_img_" + std::to_string(reinterpret_cast<uintptr_t>(&decoded)) + ".jpg";
                                FILE * f = fopen(image_path.c_str(), "wb");
                                if (f) {
                                    fwrite(decoded.data(), 1, decoded.size(), f);
                                    fclose(f);
                                    RKLLM_INF("base64 image decoded: %zu bytes → %s\n", decoded.size(), image_path.c_str());
                                } else {
                                    RKLLM_LOG("failed to write temp image file\n");
                                    image_path.clear();
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Function calling
    if (body.contains("tools") && !body["tools"].empty()) {
        std::string sys_prompt;
        for (auto & msg : body["messages"]) {
            if (msg.value("role","") == "system" && msg["content"].is_string()) { sys_prompt = msg["content"]; break; }
        }
        rkllm_set_function_tools(handle_, sys_prompt.c_str(), body["tools"].dump().c_str(), "tool_response");
        RKLLM_INF("FC tools configured: %zu\n", body["tools"].size());
    }

    // Determine input type: multimodal if image present + vision encoder loaded
    bool use_multimodal = !image_path.empty() && has_vision();

    // Encode image if multimodal
    std::vector<float> image_embed;
    int n_image_tokens = 0;
    if (use_multimodal) {
        if (!encode_image(image_path, image_embed, n_image_tokens)) {
            RKLLM_LOG("image encoding failed, falling back to text-only\n");
            use_multimodal = false;
        } else {
            RKLLM_INF("image encoded: %d tokens\n", n_image_tokens);
            // RKLLM requires <image> tag in the prompt for multimodal input
            if (prompt.find("<image>") == std::string::npos) {
                prompt = "<image>" + prompt;
            }
        }
    }

    // Helper lambda to run rkllm_run (shared by streaming/non-streaming)
    auto run_inference = [&](stream_ctx & ctx) {
        RKLLMInput rkllm_input;
        memset(&rkllm_input, 0, sizeof(RKLLMInput));
        rkllm_input.role = role.c_str();
        rkllm_input.enable_thinking = false;

        if (use_multimodal) {
            rkllm_input.input_type = RKLLM_INPUT_MULTIMODAL;
            rkllm_input.multimodal_input.prompt = (char *)prompt.c_str();
#if RKLLM_API_ABI_VERSION >= 130
            rkllm_input.multimodal_input.image.image_embed = image_embed.data();
            rkllm_input.multimodal_input.image.n_image_tokens = n_image_tokens;
            rkllm_input.multimodal_input.image.n_image = 1;
            rkllm_input.multimodal_input.image.image_width = vision_width_;
            rkllm_input.multimodal_input.image.image_height = vision_height_;
            rkllm_input.multimodal_input.image.image_start   = (char *)"<image>";
            rkllm_input.multimodal_input.image.image_end     = (char *)"</image>";
            rkllm_input.multimodal_input.image.image_content = (char *)"<image>";
#else
            rkllm_input.multimodal_input.image_embed = image_embed.data();
            rkllm_input.multimodal_input.n_image_tokens = n_image_tokens;
            rkllm_input.multimodal_input.n_image = 1;
            rkllm_input.multimodal_input.image_width = vision_width_;
            rkllm_input.multimodal_input.image_height = vision_height_;
#endif
        } else {
            rkllm_input.input_type = RKLLM_INPUT_PROMPT;
            rkllm_input.prompt_input = prompt.c_str();
        }

        RKLLMInferParam infer_params;
        memset(&infer_params, 0, sizeof(RKLLMInferParam));
        infer_params.mode = RKLLM_INFER_GENERATE;
        infer_params.keep_history = 0;
#if RKLLM_API_ABI_VERSION >= 130
        infer_params.max_new_tokens = max_tokens;
#endif

        if (rkllm_run(handle_, &rkllm_input, &infer_params, &ctx) != 0) {
            ctx.error.store(true);
            ctx.cv.notify_all();
        }
    };

    // --- Non-streaming ---
    if (!stream) {
        abort_inference();
        std::lock_guard<std::mutex> lk(mutex_);
        stream_ctx ctx;
        run_inference(ctx);

        std::string full_output, chunk;
        while (ctx.pop(chunk, 500)) full_output += chunk;

        json response = {
            {"id", "chatcmpl-rkllm"}, {"object", "chat.completion"}, {"model", model_name},
            {"choices", json::array({{
                {"index", 0},
                {"message", {{"role", "assistant"}, {"content", full_output}}},
                {"finish_reason", ctx.error.load() ? "error" : "stop"},
            }})},
            {"usage", rkllm_usage_json(ctx)},
            {"timings", rkllm_timings_json(ctx)},
        };
        res->data = response.dump();
        res->content_type = "application/json";
        return res;
    }

    // --- Streaming ---
    auto ctx = std::make_shared<stream_ctx>();
    if (worker_.joinable()) worker_.join();

    // Capture image_embed by move into the thread
    auto embed_ptr = std::make_shared<std::vector<float>>(std::move(image_embed));
    int n_tokens_copy = n_image_tokens;
    bool multimodal_copy = use_multimodal;

    worker_ = std::thread([this, prompt, role, embed_ptr, n_tokens_copy, multimodal_copy, max_tokens, ctx]() {
        std::lock_guard<std::mutex> lk(mutex_);

        RKLLMInput rkllm_input;
        memset(&rkllm_input, 0, sizeof(RKLLMInput));
        rkllm_input.role = role.c_str();
        rkllm_input.enable_thinking = false;

        if (multimodal_copy) {
            rkllm_input.input_type = RKLLM_INPUT_MULTIMODAL;
            rkllm_input.multimodal_input.prompt = (char *)prompt.c_str();
#if RKLLM_API_ABI_VERSION >= 130
            rkllm_input.multimodal_input.image.image_embed = embed_ptr->data();
            rkllm_input.multimodal_input.image.n_image_tokens = n_tokens_copy;
            rkllm_input.multimodal_input.image.n_image = 1;
            rkllm_input.multimodal_input.image.image_width = vision_width_;
            rkllm_input.multimodal_input.image.image_height = vision_height_;
            rkllm_input.multimodal_input.image.image_start   = (char *)"<image>";
            rkllm_input.multimodal_input.image.image_end     = (char *)"</image>";
            rkllm_input.multimodal_input.image.image_content = (char *)"<image>";
#else
            rkllm_input.multimodal_input.image_embed = embed_ptr->data();
            rkllm_input.multimodal_input.n_image_tokens = n_tokens_copy;
            rkllm_input.multimodal_input.n_image = 1;
            rkllm_input.multimodal_input.image_width = vision_width_;
            rkllm_input.multimodal_input.image_height = vision_height_;
#endif
        } else {
            rkllm_input.input_type = RKLLM_INPUT_PROMPT;
            rkllm_input.prompt_input = prompt.c_str();
        }

        RKLLMInferParam infer_params;
        memset(&infer_params, 0, sizeof(RKLLMInferParam));
        infer_params.mode = RKLLM_INFER_GENERATE;
        infer_params.keep_history = 0;
#if RKLLM_API_ABI_VERSION >= 130
        infer_params.max_new_tokens = max_tokens;
#endif

        if (rkllm_run(handle_, &rkllm_input, &infer_params, ctx.get()) != 0) {
            ctx->error.store(true);
            ctx->cv.notify_all();
        }
        ctx->done.store(true);
        ctx->cv.notify_all();
    });

    auto stop_fn = should_stop;
    auto name = model_name_;
    auto * self = this;
    res->content_type = "text/event-stream";
    res->next = [ctx, name, stop_fn, self, timings_per_token, include_usage](std::string & chunk) -> bool {
        if (stop_fn()) {
            ctx->abort_requested.store(true);
            self->abort_inference();
            const json timings = rkllm_timings_json(*ctx);
            chunk = sse_chunk(name, "", "stop", include_usage ? json(nullptr) : timings);
            if (include_usage) chunk += sse_usage_chunk(name, rkllm_usage_json(*ctx), timings);
            chunk += sse_done();
            return false;
        }
        if (ctx->pop(chunk, 500)) {
            chunk = sse_chunk(name, chunk, "", timings_per_token ? rkllm_timings_json(*ctx) : json(nullptr));
            return true;
        }
        if (ctx->done.load() || ctx->error.load()) {
            const json timings = rkllm_timings_json(*ctx);
            chunk = sse_chunk(name, "", ctx->error.load() ? "error" : "stop",
                              include_usage ? json(nullptr) : timings);
            if (include_usage) chunk += sse_usage_chunk(name, rkllm_usage_json(*ctx), timings);
            chunk += sse_done();
            if (self->worker_.joinable()) self->worker_.join();
            return false;
        }
        chunk.clear();
        return true; // keepalive
    };
    return res;
}

#endif // GGML_USE_RKNPU2
