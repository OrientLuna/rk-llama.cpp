#pragma once

#ifdef GGML_USE_RKNPU2

#include "server-http.h"
#include <nlohmann/json.hpp>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <thread>
#include <atomic>
#include <chrono>
#include <string>
#include <vector>

// Forward declare from rkllm.h
typedef void* LLMHandle;

// Forward declare RKNN vision encoder context (defined in rkllm-instance.cpp)
struct rknn_app_context_t;

/**
 * rkllm_model_instance — in-process RKLLM backend for llama-server router.
 *
 * Supports text and multimodal (text+image) inference via librkllmrt.so.
 * Vision encoding via librknnrt.so (.rknn vision encoder).
 */
struct rkllm_model_instance {
    LLMHandle handle_ = nullptr;
    std::mutex mutex_;        // serializes rkllm_run calls
    std::thread worker_;      // current inference worker thread
    bool loaded_ = false;

    // Vision encoder (RKNN)
    rknn_app_context_t * vision_enc_ = nullptr;
    int vision_width_  = 0;
    int vision_height_ = 0;
    int vision_image_token_ = 0;
    int vision_embed_size_  = 0;
    int vision_n_output_    = 0;

    // Producer-consumer bridge (public for callback access)
    struct stream_ctx {
        std::queue<std::string> chunks;
        std::mutex mtx;
        std::condition_variable cv;
        std::atomic<bool> done{false};
        std::atomic<bool> error{false};
        std::atomic<bool> abort_requested{false};

        struct metrics_snapshot {
            int prompt_tokens = 0;
            float prompt_ms = 0;
            int generated_tokens = 0;
            float generated_ms = 0;
            bool has_perf = false;
        };

        // RKLLM only provides authoritative perf data in the finish callback.
        // Keep a lightweight live estimate for timings_per_token streaming.
        mutable std::mutex metrics_mtx;
        int generated_tokens = 0;
        bool generation_started = false;
        std::chrono::steady_clock::time_point generation_started_at{};
        metrics_snapshot perf;

        void note_generated_token();
        void set_perf(float prompt_ms, int prompt_tokens, float generated_ms, int generated_tokens);
        metrics_snapshot get_metrics() const;

        void push(std::string && chunk) {
            { std::lock_guard<std::mutex> lk(mtx); chunks.push(std::move(chunk)); }
            cv.notify_one();
        }

        bool pop(std::string & out, int timeout_ms = 1000) {
            std::unique_lock<std::mutex> lk(mtx);
            if (cv.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                    [this] { return !chunks.empty() || done || error; })) {
                if (!chunks.empty()) { out = std::move(chunks.front()); chunks.pop(); return true; }
                return false;
            }
            return false;
        }
    };

    ~rkllm_model_instance();

    // Load .rkllm model. If vision_encoder_path is non-empty, also load RKNN vision encoder.
    bool load(const std::string & model_path, int max_context_len = 2048,
              const std::string & vision_encoder_path = "");
    void unload();
    bool is_loaded() const { return loaded_; }
    bool has_vision() const { return vision_enc_ != nullptr; }

    void abort_inference();

    server_http_res_ptr chat_completion(
        const std::string & model_name,
        const std::string & body_json,
        const std::function<bool()> & should_stop
    );

private:
    std::string model_name_;

    // Encode image file → embedding vector (via RKNN vision encoder)
    // Returns true on success, fills embed vector + token count
    bool encode_image(const std::string & image_path, std::vector<float> & embed, int & n_tokens);

    static std::string sse_chunk(const std::string & model, const std::string & content, const std::string & finish_reason = "");
    static std::string sse_chunk(const std::string & model, const std::string & content,
                                 const std::string & finish_reason, const nlohmann::json & timings);
    static std::string sse_usage_chunk(const std::string & model, const nlohmann::json & usage,
                                       const nlohmann::json & timings);
    static std::string sse_done();
};

#endif // GGML_USE_RKNPU2
