# RKLLM Backend Boundary

This note records the boundary between the unified router and Rockchip's
closed RKLLM runtime. It is intentionally based on the public headers,
observed target-device behavior, and the code in this branch. It does not
assume private symbols or undocumented runtime reset behavior.

## Result

The practical integration point is a server-level backend adapter:

```text
router / model registry / /v1/models / request owner / lifecycle
                              |
                 server_model_backend
                    /                 \
          GGUF adapter              RKLLM adapter
       server_context + ggml       librkllmrt + RKNN encoder
```

The adapter boundary is now used by `server_models`. Both formats share model
registration, explicit tags, backend discovery, request ownership, unload
draining, and the HTTP response contract. They do not share the actual model
graph, tokenizer, sampler, KV cache, or scheduler.

## ABI Versions

The `main` branch and `runtime-1.3.0` branch are source-compatible through
conditional code, but their RKLLM ABIs are not interchangeable at link/runtime
level.

| Branch | Header/runtime | Relevant difference |
|---|---|---|
| `main` | 1.2.3 | `rkllm_init` takes the raw result callback; multimodal input uses the older flat fields; `RKLLMInferParam` has no per-call `max_new_tokens`. |
| `runtime-1.3.0` | 1.3.0 | `rkllm_init` takes `RKLLMCallback`; multimodal input has a nested image structure; per-call `max_new_tokens` is available; the header defines `RKLLM_API_ABI_VERSION 130`. |

The server uses `RKLLM_API_ABI_VERSION` where available and treats an absent
macro as the 1.2.3 ABI. Replacing only `librkllmrt.so` without rebuilding with
the matching header is unsupported.

## Public Runtime Boundary

The public API exposes model-instance operations such as:

- create/init and destroy;
- synchronous/asynchronous run, abort, and KV-cache clearing;
- callback results containing text, token id, optional logits/hidden states,
  and performance statistics;
- multimodal input containing caller-provided image embeddings;
- optional function-tool and cross-attention configuration.

It does not expose a process-wide RKLLM/RKNN teardown or reset operation. The
available `rkllm_destroy` and per-context `rknn_destroy` calls are therefore
not equivalent to resetting all driver/runtime state. The implementation does
not use `dlclose`, private symbols, or binary patching as a cleanup strategy.

## Target Evidence

The target is RK3588/aarch64. The validated `runtime-1.3.0` test binary used
RKLLM runtime 1.3.0 and reported RKNN driver 0.9.8. The production service on
port 8080 was left running throughout the test; all experiments used a
separate localhost port and process.

Observed in one test process with `LLAMA_SERVER_IN_PROCESS_GGUF=1`:

1. `/v1/models` listed GGUF and RKLLM entries with the same explicit `tags`
   field and distinct `backend` values (`llama` and `rkllm`).
2. GGUF loaded and generated a response with token usage.
3. GGUF was unloaded, RKLLM loaded, and RKLLM generated a streaming response
   with `usage` and `timings`.
4. RKLLM multimodal input using an RKNN vision encoder returned a description
   with `usage` and `timings`.
5. After RKLLM unload, loading in-process GGUF produced a controlled HTTP 500:

   ```text
   cannot load an in-process GGUF model after unloading RKLLM in this process;
   restart llama-server to reset the RKNN runtime
   ```

   The router process remained alive.

Without the guard, the reverse switch produced RKNN allocation failures such
as `Too many open files` and `failed to malloc npu memory`, followed by process
failure in the original experiment. A high process fd limit removes the
ordinary 1024-fd deployment failure, but does not remove the RKLLM teardown
boundary. The service template therefore sets `LimitNOFILE=65536`, while the
model manager still serializes in-process NPU residency.

The focused reproduction is [`deploy/smoke-unified.sh`](../deploy/smoke-unified.sh).

## Why Full GGML Integration Is Not Yet Feasible

`server_context` owns a GGUF-backed `llama_model`, `llama_context`, GGML
backend scheduler, slots, sampler chain, KV cache, and `mtmd` path. RKLLM owns
an opaque `LLMHandle` and accepts a different prompt/callback protocol. The
runtime does not provide a way to import a `.rkllm` graph into GGML or to make
GGML own the RKLLM KV cache and scheduler.

Consequently, putting RKLLM behind a lower-level GGML backend would require at
least one of the following unsupported assumptions:

- reconstructing the opaque `.rkllm` graph and memory layout;
- replacing the RKLLM tokenizer/sampler/KV-cache pipeline with llama.cpp
  equivalents without a runtime API for those objects;
- finding private process-reset symbols or changing the closed runtime;
- accepting an ABI that changes independently of this repository.

Those approaches are not a maintainable default. The current adapter keeps the
shared, testable behavior above the ABI boundary and leaves RKLLM-specific
translation below it. A true lower-level backend can be reconsidered if
Rockchip exposes a documented reset/lifecycle API and tensor/context ownership
contract.

## Operational Contract

- Leave `LLAMA_SERVER_IN_PROCESS_GGUF` unset for the default GGUF child-process
  isolation.
- With it enabled, treat NPU residency as serialized even when `models-max` is
  greater than one.
- `GGUF -> RKLLM` is validated on the 1.3.0 target runtime.
- `RKLLM -> GGUF` requires a process restart on that runtime; the router returns
  an explicit error instead of attempting an unsafe reinitialization.
- The 1.2.3 path has source-level ABI coverage, but the exact runtime/model
  combination still requires separate hardware validation on the `main`
  branch.
