# rk-llama.cpp

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

[简体中文](./README.md) | **English**

A fork of [rk-llama.cpp](https://github.com/invisiofficial/rk-llama.cpp) that aims to support both `*.rkllm` and `*.gguf` models under a single runtime — the former offers better NPU inference optimization, the latter a faster model-adaptation pace. There is nothing technically novel here. The repo is mostly the product of vibe-coding, so some of the docs may not read like a human wrote them; the author will keep refining them, and discussion is welcome.

> Note: `rk-llama.cpp` is itself a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp). This fork adds a **Rockchip RKLLM in-process backend** and **multimodal vision support** to `llama-server`, targeting Rockchip NPUs such as the **RK3588**.

> This README covers only the additions in this fork (the RKLLM backend + multimodal path). For the upstream Rockchip NPU GGML backend (hybrid quantization on `.gguf` models), see [ggml/src/ggml-rknpu2/README.md](./ggml/src/ggml-rknpu2/README.md).

### Tested environment

| | |
|---|---|
| **SoC** | Rockchip **RK3588** (4× Cortex-A76 + 4× Cortex-A55, 3 NPU cores), `aarch64` |
| **OS** | Ubuntu 22.04 (modified from the image by Forlinx) |
| **Build** | Native on-device with `gcc/g++ 11.4`, `cmake 3.22` |
| **RKNN driver** | `0.9.6+` (NPU runtime `librknnrt.so`) |
| **RKLLM runtime** | `1.2.3` (`main`) and `1.3.0` (`runtime-1.3.0`) |

Other Rockchip SoCs (e.g. RK3576) and other RKNN/RKLLM versions are **not yet verified** (see TODO).

#### RKLLM runtime and branch

`.rkllm` models generally need the `rkllm-toolkit`/runtime version that produced them. The two branches ship matching headers, libraries, and service ABIs:

| Branch | `librkllmrt.so` | Service ABI | Intended models |
|---|---|---|---|
| `main` | `1.2.3` | 1.2.3 | `.rkllm` models produced by toolkit 1.2.x |
| `runtime-1.3.0` | `1.3.0` | 1.3.0 | `.rkllm` models produced by toolkit 1.3.x |

Both ABIs support the RKLLM `usage/timings` implementation, but the binary must be built with the matching branch and header. Do not replace only the `.so` and keep a binary built for the other ABI. Use [`deploy/fetch-rkllm-runtime.sh`](./deploy/fetch-rkllm-runtime.sh) to inspect or switch runtimes; changing ABI requires a rebuild.

### Tested models

The following model files have been verified to load and run on the tested environment above. Both were converted for the RK3588 target (`rkllm-toolkit` for `.rkllm`, the standard toolchain for `.gguf`).

| Model | File(s) | Backend | Multimodal | Notes |
|---|---|---|---|---|
| **Qwen3-VL 2B Instruct** | `qwen3-vl-2b-instruct_w8a8_rk3588.rkllm` (2.3 GB) + `qwen3-vl-2b_vision_rk3588.rknn` (812 MB) | `rkllm` | ✅ vision | W8A8 quant; in-process backend |
| **Qwen3.5 0.8B** | `Qwen3.5-0.8B-Q4_K_M.gguf` (508 MB) + `mmproj-Qwen3.5-0.8B-F16.gguf` (196 MB) | `llama` (NPU) | ✅ projector | Q4_K_M; runs on the NPU via `ggml-rknpu2`, ~21 tok/s |

Tested flows: cold load, standard router model switch (unload → load), chat completion (text + Chinese + arithmetic), and the router API (`/models/load`, `/v1/chat/completions`). The experimental in-process path has additional switch boundaries described below. More models will be added as they are validated — broader coverage is tracked in the TODO section.

---

## Quick start

Don't want to read the whole thing? Five steps to get running (assuming you're on an RK3588 and your model files are in place):

```sh
# 1. Build (~5 min)
cd rk-llama.cpp && mkdir build && cd build
cmake .. -DLLAMA_RKNPU2=ON && make -j$(nproc) llama-server

# 2. Create your model config from the template, point model/mmproj at your files
cd ..
cp deploy/models.ini.example deploy/models.ini
#   edit deploy/models.ini — at minimum fix the model path in one [section]

# 3. Raise the file-descriptor limit
ulimit -n 65536

# 4. Start the server (replace <your-model-id> with the section name from models.ini)
./build/bin/llama-server --models-preset deploy/models.ini \
    --default-model <your-model-id> --models-max 1 \
    --host 0.0.0.0 --port 8080 -c 2048

# 5. Test it
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

Works? For finer configuration (multiple models, autostart on boot, etc.), see the [Run](#run) and [Deployment](#deployment) sections below. Stuck? See [Troubleshooting](#troubleshooting).

---

## What this fork adds

The upstream rk-llama.cpp accelerates **`.gguf`** models on the NPU via a custom GGML backend (`ggml-rknpu2`). This fork adds two things on top:

1. **RKLLM in-process backend** — runs native Rockchip **`.rkllm`** model files inside the `llama-server` process through `librkllmrt.so`. Unlike the standard router path (which spawns a child HTTP server per model), the RKLLM backend loads and runs the model in-process — no subprocess, no extra port.
2. **Multimodal vision support** — an RKNN **`.rknn`** vision encoder runs on `librknnrt.so`, produces image embeddings, and feeds them to the RKLLM backbone via `RKLLM_INPUT_MULTIMODAL`. Images can be supplied as a file path, a `file://` URI, or a base64 `data:` URI.

The original `.gguf` + NPU path still works (see the backend README linked above).

### Experimental single-process model manager

By default, the router still starts a separate `llama-server` child process for
GGUF models, while RKLLM runs in the router process. Set the following variable
to load GGUF through an in-process `server_context` as well. One process then
shares model registration, `/v1/models`, request routing, unload, and switching
for both formats:

```sh
LLAMA_SERVER_IN_PROCESS_GGUF=1 \
  ./build/bin/llama-server --models-preset deploy/models.ini \
  --default-model my-gguf-model --models-max 1 \
  --host 0.0.0.0 --port 8080 -c 2048
```

This is currently an opt-in POC; the default remains the isolated GGUF child
process. On the target device with RKLLM runtime 1.3.0, `GGUF -> RKLLM` works in
one process. After RKLLM is unloaded, the runtime may retain process-level RKNN
state, so `RKLLM -> GGUF` returns HTTP 500 and asks for a `llama-server` restart
instead of risking a crash. `--models-max` limits active entries in the model
manager but does not guarantee that multiple NPU backends can remain resident;
NPU models are serialized and the old instance is unloaded before switching.

The router now targets a minimal `server_model_backend` management interface,
but RKLLM inference, vision encoding, and ABI handling are still implemented by
a dedicated adapter. It is not yet a general GGML compute backend.
See [`docs/rkllm-backend-boundary.md`](./docs/rkllm-backend-boundary.md) for the
ABI, runtime-resource evidence, and the detailed assessment of the higher
integration target.

### Supported model formats

| Format | Role | Runtime |
|---|---|---|
| `.rkllm` | Native Rockchip LLM (in-process backend) | `librkllmrt.so` |
| `.rknn` | Vision encoder for multimodal input | `librknnrt.so` |
| `.gguf` | Standard llama.cpp / NPU-accelerated backbone | `librknpu2` backend |

---

## Build

> Build **natively on the Rockchip device** (e.g. RK3588). The target is `aarch64` — don't cross-compile from x86 unless you have a matching toolchain and sysroot. (If you're an expert at this, feel free to try.)

### Prerequisites

A standard Ubuntu/Debian toolchain on the board:

```sh
sudo apt-get install -y build-essential cmake git
```

The Rockchip **RKNN runtime** (`librknnrt.so`) and **RKLLM runtime** (`librkllmrt.so`), plus their SDK headers, are **vendored** under `ggml/src/ggml-rknpu2/libs/` — no download step is needed. These are Rockchip's proprietary NPU runtimes (see [NOTICE.md](./NOTICE.md)); Rockchip does not provide a GGUF backend. The `.gguf`-on-NPU support is implemented by this repo's `ggml-rknpu2` backend, which calls into `librknnrt.so`. `llama-server` links both runtimes directly, so at runtime the loader must find them (see Troubleshooting below).

> (This paragraph was written by AI; it may not be entirely accurate, but it works for now, so I'm leaving it.)

### Configure and compile

```sh
cd rk-llama.cpp
mkdir build && cd build
cmake .. -DLLAMA_RKNPU2=ON
make -j$(nproc) llama-server
```

A full native build of `llama-server` on an RK3588 takes roughly 5 minutes with `-j8`. CMake should print `-- RKLLM in-process backend enabled for llama-server` during configure — that line confirms the RKLLM path is wired in. The binary lands at `build/bin/llama-server`.

Verify it links the NPU runtimes:

```sh
ldd build/bin/llama-server | grep -E 'rknn|rkllm'
# expect: librkllmrt.so => .../libs/librkllmrt.so
#         librknnrt.so => .../libs/librknnrt.so
```

### Raise the open-file limit before running

The NPU runtime opens many file descriptors. Raise the limit in the shell you launch from (or permanently via `/etc/security/limits.conf`):

```sh
ulimit -n 65536
```

### A note on the Rockchip runtimes

The two vendored `.so` files are Rockchip's proprietary NPU runtimes (see the Tested environment table above and [NOTICE.md](./NOTICE.md)):

- `librknnrt.so` — the **RKNN runtime** (NPU compute). Called by this repo's `ggml-rknpu2` backend (which runs `.gguf` models on the NPU) **and** by the RKNN vision encoder. Rockchip does not ship a "GGUF backend"; that backend is code in this repo targeting the RKNN runtime.
- `librkllmrt.so` — the **RKLLM runtime**, used by the in-process RKLLM backend to run native `.rkllm` models.

### Troubleshooting

- **`error while loading shared libraries: librknnrt.so: cannot open shared object file`** — the loader can't find the runtimes. Either run from the repo root (the build uses RPATH relative to the binary), or export `LD_LIBRARY_PATH=/path/to/rk-llama.cpp/ggml/src/ggml-rknpu2/libs`.
- **CMake error `Cannot find source file: models/afmoe.cpp`** — the source tree is incomplete; re-sync/checkout `src/models/` (it must contain ~114 `.cpp` files). Do not gitignore or exclude `src/models/`.

---

## Run

### 1. Place your model files

Pick a directory to hold model weights. This is **your choice** — there is no fixed location. Two common layouts:

```sh
# Option A: keep models inside the repo (simple, good for development)
mkdir -p models
#   models/qwen2.5-0.5b-q4_k_m.gguf
#   models/mmproj-qwen2.5-f16.gguf
#   models/qwen2.5-0.5b-w4a16.rkllm      (RKLLM-converted)
#   models/qwen2.5-vision-encoder.rknn   (RKNN vision encoder)

# Option B: keep models outside the repo (good for production / shared storage)
mkdir -p /opt/models   # or /data/models, an SD card mount, an NVMe path, etc.
#   /opt/models/...
```

What you need per model:

| Backend | LLM file | Vision (optional) | Convert with |
|---|---|---|---|
| GGUF (NPU-accelerated) | `*.gguf` (quantized, e.g. Q4_K_M) | `mmproj-*.gguf` | standard llama.cpp quantization |
| RKLLM (in-process) | `*.rkllm` | `*.rknn` vision encoder | Rockchip `rkllm-toolkit` |

The NPU-accelerated GGUF path works with any standard quantized `.gguf`; the backend re-quantizes it on the fly to native NPU pipelines (W16A16 / W8A8 / W4A4). For RKLLM, the `.rkllm` file must be produced by Rockchip's `rkllm-toolkit` for the RK3588 target.

### 2. Edit `models.ini`

The router's model registry is an INI file: each `[section]` is one model, and keys configure it. Start from the shipped template and point the `model` / `mmproj` keys at the files you placed:

```sh
cp deploy/models.ini.example deploy/models.ini
# then edit deploy/models.ini
```

> `deploy/models.ini` is gitignored — it's your local config. The tracked template is [`deploy/models.ini.example`](./deploy/models.ini.example). This keeps re-syncing the repo from clobbering your edited file.

Paths can be **relative** (resolved from the directory you launch `llama-server` in) or **absolute**:

```ini
version = 1

# The special [*] section applies defaults to every model below.
[*]
c = 2048
jinja = true
reasoning = off

# --- GGUF model (NPU-accelerated via the ggml-rknpu2 backend) ---
[my-gguf-model]
model = models/qwen2.5-0.5b-q4_k_m.gguf      # relative path
mmproj = models/mmproj-qwen2.5-f16.gguf       # multimodal projector (optional)
load-on-startup = true                         # preload when the server starts

# --- RKLLM model (in-process backend) ---
[my-rkllm-model]
backend = rkllm                                # <-- selects the in-process RKLLM path
model = /opt/models/qwen2.5-0.5b-w4a16.rkllm   # absolute path is fine too
mmproj = /opt/models/qwen2.5-vision-encoder.rknn   # .rknn vision encoder (optional)
c = 2048
```

The section name (`[my-gguf-model]`) becomes the model's ID in the API — name it whatever you like. Add as many sections as you want; `--models-max N` caps how many can be loaded at once.

`models.ini` remains the source of model registration and runtime configuration. `/v1/models` is a read-only discovery view of models already registered in the INI; it does not replace the INI or create model entries. Put `tags` in each GGUF/RKLLM section when you want them shown in the WebUI or API. These are explicit registry tags (the same field as single-model `--tags`), not tags inferred from GGUF metadata, so both formats use the same INI field.

**INI key reference:**

| Key | Meaning |
|---|---|
| `model` | Path to the model file (`.rkllm` or `.gguf`) |
| `backend = rkllm` | Load in-process via `librkllmrt.so`. **Omit** for the GGUF / subprocess path. |
| `mmproj` | RKLLM backend: path to the **`.rknn` vision encoder**. GGUF backend: the standard `mmproj-*.gguf` projector. Optional. |
| `tags` | Optional comma-separated WebUI/API tags shared by GGUF and RKLLM, e.g. `0.8B,gguf` or `0.8B,rkllm,rknn`; the server reads them from the INI and exposes them through `/v1/models`. |
| `c` | Max context length (default 2048 for RKLLM) |
| `load-on-startup` | `true` to **preload** this model when the server starts |
| `jinja`, `reasoning` | Template / reasoning-format toggles |

**Router CLI flags** (`common/common.h`, `common/arg.cpp`):

| Flag | Env var | Default | Meaning |
|---|---|---|---|
| `--models-preset PATH` | `LLAMA_ARG_MODELS_PRESET` | disabled | INI file of model presets |
| `--models-dir PATH` | `LLAMA_ARG_MODELS_DIR` | disabled | Directory of `.gguf` models to scan |
| `--models-max N` | `LLAMA_ARG_MODELS_MAX` | `4` (`0` = unlimited) | Max models loaded simultaneously |
| `--default-model NAME` | `LLAMA_ARG_DEFAULT_MODEL` | none | Model used when a request omits `model:` |
| `--models-autoload` / `--no-models-autoload` | `LLAMA_ARG_MODELS_AUTOLOAD` | enabled | Auto-load an unloaded model on demand when a request names it |

Note the distinction between the three "load" mechanisms:

- **`load-on-startup`** (INI) — preloads a model at server boot.
- **`--no-models-autoload`** (CLI) — blocks *on-demand* loading; a request for an unloaded model errors instead of triggering a load. (It does **not** affect `load-on-startup`.)
- **`--default-model`** (CLI) — which model a request resolves to when it doesn't specify one.

### 3. Start the server (router mode)

With `models.ini` configured, launch `llama-server` with **no `-m`** and point it at the preset:

```sh
./build/bin/llama-server \
    --models-preset deploy/models.ini \
    --default-model my-rkllm-model \
    --models-max 1 \
    --host 0.0.0.0 --port 8080 -c 2048 --reasoning off
```

The web UI is then at `http://<device-ip>:8080`. Models are loaded via the `/models/load` API, the web UI, or `load-on-startup = true`. With `--default-model` set, chat requests that omit the `model` field resolve to that model.

**Router API quick reference** (prefix none — these are not under `/v1/` except where shown):

| Method & path | Body / query | Effect |
|---|---|---|
| `GET /v1/models` | — | List all models + load status |
| `POST /models/load` | `{"model":"<id>"}` | Load (or queue) a model |
| `POST /models/unload` | `{"model":"<id>"}` | Unload a model |
| `POST /v1/chat/completions` | OpenAI body (with `model`) | Chat completion |

### 4. Multimodal (vision) input

Send images in OpenAI vision-API style — a `content` array with `type: "image_url"` entries. Accepted image sources:

- Absolute path: `/abs/path/image.jpg`
- `file:///abs/path/image.jpg`
- Data URI: `data:image/jpeg;base64,<base64 data>`

If the prompt lacks an `<image>` tag, one is injected automatically before the image embeddings.

### Single-model mode (no preset)

For a quick test you can skip the preset and pass one model directly:

```sh
./build/bin/llama-server -m models/my-model.gguf --host 0.0.0.0 --port 8080 -c 2048
```

(The in-process RKLLM path is only reached through the router + `backend = rkllm` preset — there is no single-model `-m` flag for `.rkllm` files.)

---

## Deployment

For running `llama-server` as a background service on boot, a systemd unit and installer are provided under [`deploy/`](./deploy/).

### 1. Edit the systemd unit

Open [`deploy/systemd/llama-server.service`](./deploy/systemd/llama-server.service) and replace the placeholders with your setup:

```ini
[Service]
User=<your-user>                      # the account that owns the checkout
Group=<your-user>
WorkingDirectory=/opt/rk-llama.cpp    # absolute path to the project
Environment=LD_LIBRARY_PATH=/opt/rk-llama.cpp/ggml/src/ggml-rknpu2/libs
ExecStart=/opt/rk-llama.cpp/build/bin/llama-server \
    --models-preset /opt/rk-llama.cpp/deploy/models.ini \
    --default-model <model-id> \
    --models-max 1 \
    --host 0.0.0.0 --port 8080 -c 2048 --reasoning off
LimitNOFILE=65536
```

Key points:

- `User=` / `Group=` — the account that owns the checkout **and** has read access to the model files under `/userdata/models/` (or wherever you placed them).
- `WorkingDirectory` + the two absolute paths in `ExecStart` — point them at your checkout.
- `LD_LIBRARY_PATH` — must include the `libs/` dir so the loader finds `librknnrt.so` / `librkllmrt.so`. The unit also sets `LimitNOFILE=65536` so you don't need the manual `ulimit` for the service.
- `--models-preset` — points at your edited `models.ini` (copy it from `deploy/models.ini.example` first; see the Run section).
- `--default-model <model-id>` — optional; the section name from `models.ini` to serve when a request omits `model:`. Remove the line if you want every request to name a model explicitly.
- `LLAMA_SERVER_IN_PROCESS_GGUF=1` — optional experimental switch; enables the in-process GGUF adapter alongside `backend = rkllm`. The RKLLM → GGUF restart boundary is described above; production deployments should leave it unset by default.

### 2. Install and enable

[`deploy/install.sh`](./deploy/install.sh) copies the unit into `/etc/systemd/system/` and enables autostart on boot:

```sh
# after editing the unit:
sudo bash deploy/install.sh
sudo systemctl start llama-server
sudo journalctl -u llama-server -f   # tail logs
```

The API is then at `http://<device-ip>:8080`.

### Other deployment notes

- **Working without systemd** — for development you can launch directly: `cd /opt/rk-llama.cpp && ulimit -n 65536 && ./build/bin/llama-server --models-preset deploy/models.ini ...`.
- **Models on a separate partition** — common on boards where `/userdata` is a large eMMC/NVMe partition. Just point `models.ini` paths (e.g. `/userdata/models/...`) at it; the service user needs read access.
- **Single-process smoke test** — with `LLAMA_SERVER_IN_PROCESS_GGUF=1` on a test service, run [`deploy/smoke-unified.sh`](./deploy/smoke-unified.sh) to verify `/v1/models`, shared tags, RKLLM streaming `usage/timings`, and the known switch boundary: `GGUF_MODEL=<id> RKLLM_MODEL=<id> ./deploy/smoke-unified.sh`.

---

## Status & TODO

This is an active edge-AI integration, not a finished product. Work in progress:

- [ ] **Mixed RKNN vision encoder + GGUF LLM backbone** — the `tools/mtmd/rknn_encoder` path exists but needs integration testing so an RKNN encoder can feed a `.gguf` backbone (currently the encoder is wired to the RKLLM backbone only).
- [ ] **Broader model support** — test more `.rkllm` LLM and `.rknn` vision-encoder combinations beyond the current Qwen / Gemma set.
- [x] **Driver / SDK version compatibility** — compatibility code now covers the RKNN `0.9.6+` environment and both RKLLM 1.2.3/1.3.0 ABIs; more runtime versions still need hardware validation.
- [ ] **RK3576 configuration** — fill in the stubbed device config in `ggml/src/ggml-rknpu2/`.
- [x] **RKLLM response metrics** — the RKLLM path now exposes OpenAI-compatible `usage` and `timings`. Streaming uses live estimates and replaces them with the SDK's authoritative `RKLLMPerfStat` values at completion.

---

## Credits

This project stands on the shoulders of:

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — The ggml authors. The inference engine this is built on.
- **[rk-llama.cpp](https://github.com/invisiofficial/rk-llama.cpp)** — the Rockchip NPU GGML backend fork this project derives from.
- **Rockchip** — the RKNN & RKLLM SDK (runtime libraries and headers under `ggml/src/ggml-rknpu2/libs/`).

See [NOTICE.md](./NOTICE.md) for third-party attribution and licensing notes.

---

## License

MIT — see [LICENSE](./LICENSE).

The vendored Rockchip RKNN / RKLLM SDK files (headers `rknn_api.h`, `rknn_custom_op.h`, `rknn_matmul_api.h`, `rkllm.h` and the `librknnrt.so` / `librkllmrt.so` binaries) are **proprietary to Rockchip** and governed by their own terms, **not** the MIT license of this repository. See [NOTICE.md](./NOTICE.md).
