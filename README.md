# rk-llama.cpp

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

**简体中文** | [English](./README.en.md)

本项目是[rk-llama.cpp](https://github.com/invisiofficial/rk-llama.cpp) 的分支，希望使用一套运行时,同时提供对*.rkllm以及*.gguf格式的模型兼容，前者提供更好的NPU推理优化，后者提供更快的模型适配节奏，技术上没有任何创新性。仓库主要依靠vibe-coding完成，所以有些文档不太说人话，后续作者会逐步完善，也欢迎各位交流。

> 注：`rk-llama.cpp`本身是 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的分支），在 `llama-server` 中新增了**瑞芯微 RKLLM 进程内后端**与**多模态视觉支持**，面向 RK3588 等瑞芯微 NPU。

> 本 README 仅介绍本分支新增的内容（RKLLM 后端 + 多模态通路）。关于上游的 Rockchip NPU GGML 后端（对 `.gguf` 模型做混合量化），请参阅 [ggml/src/ggml-rknpu2/README.md](./ggml/src/ggml-rknpu2/README.md)。

### 已测试环境

| | |
|---|---|
| **SoC** | 瑞芯微 **RK3588**（4× Cortex-A76 + 4× Cortex-A55，3 个 NPU 核心），`aarch64` |
| **操作系统** | Ubuntu 22.04（基于Forlinx飞凌嵌入式提供的镜像修改） |
| **编译** | 板上原生编译，`gcc/g++ 11.4`、`cmake 3.22` |
| **RKNN 驱动** | `0.9.6+` / `0.9.8`（NPU 运行时 `librknnrt.so`，版本随板子内核驱动） |
| **RKLLM 运行时** | `1.2.3`（`main` 分支内置）与 `1.3.0`（`runtime-1.3.0` 分支内置）均验证通过 |

其它瑞芯微 SoC（如 RK3576）**尚未验证**（见 TODO）。

#### RKLLM 运行时版本与分支

`.rkllm` 模型与 `rkllm-toolkit`/运行时**版本锁定**：toolkit 1.3.0 转换的模型需要 `librkllmrt.so` 1.3.0。本仓库用两个分支分别内置对应版本的头文件与动态库：

| 分支 | 内置 `librkllmrt.so` | 头文件 | 适用模型 |
|---|---|---|---|
| **`main`**（默认） | `1.2.3` | 1.2.3 | toolkit 1.2.x 转换的 `.rkllm` |
| **`runtime-1.3.0`** | `1.3.0` | 1.3.0 | toolkit 1.3.x 转换的 `.rkllm`（**同时向后兼容 1.2.x 模型**） |

两套 ABI 都支持 RKLLM 的 OpenAI 兼容 `usage` / `timings`：非流式响应返回最终值，SSE 在生成期间可返回实时估计，并在结束时使用 SDK 的 `RKLLMPerfStat`。编译时必须使用与动态库匹配的分支、头文件和二进制；不能只替换 `.so`。1.2.3 的 `RKLLMInferParam` 没有请求级 `max_new_tokens`，所以该分支使用初始化时的 512 上限；1.3.0 支持用请求中的 `max_tokens` / `max_completion_tokens` 覆盖它。

> 切换分支：`git checkout runtime-1.3.0`，然后重新编译。也可用 [`deploy/fetch-rkllm-runtime.sh`](./deploy/fetch-rkllm-runtime.sh) 单独替换 `.so` 而不切分支（见下文）。

### 已测试模型

下列模型文件已在上述环境验证可正常加载和运行。

**RKLLM 后端**（`.rkllm` + `.rknn` 视觉编码器，`backend = rkllm`）：

| 模型 | 文件 | 多模态 | 备注 |
|---|---|---|---|
| **Qwen3-VL 2B Instruct** | `qwen3-vl-2b-instruct_w8a8_rk3588.rkllm`（2.3 GB）+ `qwen3-vl-2b_vision_rk3588.rknn`（812 MB） | ✅ | W8A8；toolkit 1.2.3 |
| **Qwen3.5 0.8B**（VL） | `Qwen3.5-0.8B_w8a8_rk3588.rkllm`（1.3 GB）+ `Qwen3.5-0.8B_vision_rk3588.rknn`（209 MB） | ✅ | W8A8；toolkit 1.3.0，需 `runtime-1.3.0` 分支 |

**GGUF / NPU 后端**（标准量化 `.gguf`，经 `ggml-rknpu2` 在 NPU 上运行）：

| 模型 | 架构 | 文件 | 备注 |
|---|---|---|---|
| **Qwen3.5 0.8B** | qwen3 | `Qwen3.5-0.8B-Q4_K_M.gguf`（508 MB） | Q4_K_M，约 21 tok/s |
| **Qwen2.5 0.5B** | qwen2 | `qwen2.5-0.5b-instruct-q8_0.gguf`（645 MB） | Q8_0，加载 4s |
| **Gemma-3 1B** | gemma3 | `gemma-3-1b-it-Q8_0.gguf`（1020 MB） | Q8_0，加载 10s |
| **Llama-3.2 1B** | llama | `Llama-3.2-1B-Instruct-Q8_0.gguf`（1.3 GB） | Q8_0，加载 11s |

GGUF 后端对架构无限制——任意标准量化 `.gguf` 即可（后端即时重新量化为 NPU 原生管线 W16A16/W8A8/W4A4）。已验证 qwen2 / qwen3 / gemma3 / llama 四种架构均正常。

已验证的流程：冷启动加载、模型切换（卸载→加载）、纯文本聊天、多模态图片识别，以及 router API（`/models/load`、`/models/unload`、`/v1/chat/completions`）。更多模型将在验证后补充（见 TODO）。

---

## 快速开始

不想看长篇说明？五步跑起来（假设你已在 RK3588 上，且模型文件已就位）：

```sh
# 1. 编译（约 5 分钟）
cd rk-llama.cpp && mkdir build && cd build
cmake .. -DLLAMA_RKNPU2=ON && make -j$(nproc) llama-server

# 2. 从模板创建你的模型配置，把 model/mmproj 路径改成你的实际文件
cd ..
cp deploy/models.ini.example deploy/models.ini
#   编辑 deploy/models.ini，至少改一个 [section] 里的 model 路径

# 3. 调高文件描述符上限
ulimit -n 65536

# 4. 启动服务（把 <your-model-id> 换成 models.ini 里的小节名）
./build/bin/llama-server --models-preset deploy/models.ini \
    --default-model <your-model-id> --models-max 1 \
    --host 0.0.0.0 --port 8080 -c 2048

# 5. 测试一下
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}]}' 
```

跑通了？后续可看下面的 [运行](#运行) 和 [部署](#部署) 章节做更细的配置（多模型、开机自启等）。卡住了？看 [故障排查](#故障排查)。

---

## 本分支新增了什么

上游 rk-llama.cpp 通过自定义 GGML 后端（`ggml-rknpu2`）在 NPU 上加速 **`.gguf`** 模型。本分支在此基础上新增了两项能力：

1. **RKLLM 进程内后端** —— 通过 `librkllmrt.so` 在 `llama-server` 进程内直接运行瑞芯微原生的 **`.rkllm`** 模型文件。与标准的 router 路径（每个模型派生一个子进程 HTTP 服务）不同，RKLLM 后端在进程内加载并运行模型，无子进程、无额外端口。
2. **多模态视觉支持** —— RKNN **`.rknn`** 视觉编码器运行在 `librknnrt.so` 上，生成图像嵌入并通过 `RKLLM_INPUT_MULTIMODAL` 送入 RKLLM 主干网络。图像可通过文件路径、`file://` URI 或 base64 `data:` URI 三种方式传入。

原有的 `.gguf` + NPU 通路保持不变（见上方链接的后端 README）。

### 支持的模型格式

| 格式 | 用途 | 运行时 |
|---|---|---|
| `.rkllm` | 瑞芯微原生 LLM（进程内后端） | `librkllmrt.so` |
| `.rknn` | 多模态输入的视觉编码器 | `librknnrt.so` |
| `.gguf` | 标准 llama.cpp / NPU 加速主干 | `librknpu2` 后端 |

---

## 编译

> 建议**在瑞芯微板子上原生编译**（如 RK3588）。目标是 `aarch64`——除非你有匹配的工具链和 sysroot，否则不要从 x86 交叉编译。（如果你是这方面的专家可以自行尝试）

### 前置依赖

板子上需要标准的 Ubuntu/Debian 工具链：

```sh
sudo apt-get install -y build-essential cmake git
```

瑞芯微的 **RKNN 运行时**（`librknnrt.so`）与 **RKLLM 运行时**（`librkllmrt.so`）及 SDK 头文件已**内置**在 `ggml/src/ggml-rknpu2/libs/` 下，无需额外下载。这两个是瑞芯微的专有 NPU 运行时（见 [NOTICE.md](./NOTICE.md)）；瑞芯微**不**提供 GGUF 后端——`.gguf` 在 NPU 上运行的能力由本仓库的 `ggml-rknpu2` 后端实现，它调用 `librknnrt.so`。`llama-server` 会直接链接这两个运行时，因此运行时加载器必须能找到它们（见下方故障排查）

> 另外，这块表述是AI写的，我感觉好像不太对，但是反正目前还能用就先不管了）。

### 配置与编译

```sh
cd rk-llama.cpp
mkdir build && cd build
cmake .. -DLLAMA_RKNPU2=ON
make -j$(nproc) llama-server
```

在 RK3588 上以 `-j8` 全量原生编译 `llama-server` 大约需要 5 分钟。配置阶段应打印 `-- RKLLM in-process backend enabled for llama-server`——这一行确认 RKLLM 路径已接入。产物位于 `build/bin/llama-server`。

验证已链接 NPU 运行时：

```sh
ldd build/bin/llama-server | grep -E 'rknn|rkllm'
# 期望: librkllmrt.so => .../libs/librkllmrt.so
#       librknnrt.so => .../libs/librknnrt.so
```

### 运行前调高文件描述符上限

NPU 运行时会打开大量文件描述符。请在启动它的 shell 中调高上限（或通过 `/etc/security/limits.conf` 永久设置）：

```sh
ulimit -n 65536
```

### 关于瑞芯微运行时的说明

内置的两个 `.so` 文件是瑞芯微专有的 NPU 运行时（见上方的"已测试环境"表格及 [NOTICE.md](./NOTICE.md)）：

- `librknnrt.so` —— **RKNN 运行时**（NPU 计算）。被本仓库的 `ggml-rknpu2` 后端（在 NPU 上运行 `.gguf` 模型）**和** RKNN 视觉编码器调用。瑞芯微并不提供"GGUF 后端"——该后端是本仓库中针对 RKNN 运行时编写的代码。
- `librkllmrt.so` —— **RKLLM 运行时**，用于进程内 RKLLM 后端运行原生 `.rkllm` 模型。**版本与 `.rkllm` 模型锁定**：见上方的"RKLLM 运行时版本与分支"。

**切换 RKLLM 运行时版本**（当你的 `.rkllm` 模型需要不同版本时）：

```sh
# 方式一:切到对应分支(main=1.2.3 / runtime-1.3.0=1.3.0),重新编译
git checkout runtime-1.3.0 && cd build && make -j$(nproc) llama-server

# 方式二:用脚本单独下载替换 .so(不切分支)
./deploy/fetch-rkllm-runtime.sh --list       # 列出可用版本
./deploy/fetch-rkllm-runtime.sh --current    # 查看当前版本
./deploy/fetch-rkllm-runtime.sh 1.3.0        # 下载并替换(自动备份+校验)
```

> ⚠️ 跨大版本切换（如 1.2.3 ↔ 1.3.0）涉及头文件 ABI 变化，**必须切到对应分支并重新编译**；仅用脚本换 `.so` 而不换头文件会导致崩溃。小版本（如 1.2.2↔1.2.3）一般可直接换 `.so`。

### 故障排查

- **`error while loading shared libraries: librknnrt.so: cannot open shared object file`** —— 加载器找不到运行时。从仓库根目录运行即可（构建已用相对二进制的 RPATH），或导出 `LD_LIBRARY_PATH=/path/to/rk-llama.cpp/ggml/src/ggml-rknpu2/libs`。
- **CMake 报错 `Cannot find source file: models/afmoe.cpp`** —— 源码树不完整；重新同步/检出 `src/models/`（应包含约 114 个 `.cpp`）。**不要** gitignore 或排除 `src/models/`。

---

## 运行

### 1. 放置模型文件

选一个目录存放模型权重，**位置由你决定**，没有固定路径。两种常见布局：

```sh
# 方式 A：放在仓库内（简单，适合开发）
mkdir -p models
#   models/qwen2.5-0.5b-q4_k_m.gguf
#   models/mmproj-qwen2.5-f16.gguf
#   models/qwen2.5-0.5b-w4a16.rkllm      （RKLLM 转换）
#   models/qwen2.5-vision-encoder.rknn   （RKNN 视觉编码器）

# 方式 B：放在仓库外（适合生产 / 共享存储）
mkdir -p /opt/models   # 或 /data/models、SD 卡挂载点、NVMe 路径等
#   /opt/models/...
```

每个模型需要哪些文件：

| 后端 | LLM 文件 | 视觉（可选） | 转换工具 |
|---|---|---|---|
| GGUF（NPU 加速） | `*.gguf`（量化版，如 Q4_K_M） | `mmproj-*.gguf` | 标准 llama.cpp 量化 |
| RKLLM（进程内） | `*.rkllm` | `*.rknn` 视觉编码器 | 瑞芯微 `rkllm-toolkit` |

NPU 加速的 GGUF 路径兼容任意标准量化 `.gguf`，后端会即时重新量化为原生 NPU 管线（W16A16 / W8A8 / W4A4）。RKLLM 的 `.rkllm` 文件必须由瑞芯微 `rkllm-toolkit` 针对 RK3588 目标生成。

### 2. 编辑 `models.ini`

router 的模型注册表是一个 INI 文件：每个 `[小节]` 对应一个模型，键值对用于配置。从内置模板复制一份，再把 `model` / `mmproj` 键指向你放置的文件：

```sh
cp deploy/models.ini.example deploy/models.ini
# 然后编辑 deploy/models.ini
```

> `deploy/models.ini` 已被 gitignore——它是你的本地配置。仓库跟踪的模板是 [`deploy/models.ini.example`](./deploy/models.ini.example)。这样重新同步仓库时不会覆盖你编辑过的文件。

路径支持**相对路径**（相对于启动 `llama-server` 的工作目录）和**绝对路径**：

```ini
version = 1

# 特殊的 [*] 小节为下方所有模型设置默认值。
[*]
c = 2048
jinja = true
reasoning = off

# --- GGUF 模型（通过 ggml-rknpu2 后端做 NPU 加速）---
[my-gguf-model]
model = models/qwen2.5-0.5b-q4_k_m.gguf      # 相对路径
mmproj = models/mmproj-qwen2.5-f16.gguf       # 多模态投影器（可选）
load-on-startup = true                         # 服务启动时预加载

# --- RKLLM 模型（进程内后端）---
[my-rkllm-model]
backend = rkllm                                # <-- 选择进程内 RKLLM 路径
model = /opt/models/qwen2.5-0.5b-w4a16.rkllm   # 绝对路径也可以
mmproj = /opt/models/qwen2.5-vision-encoder.rknn   # .rknn 视觉编码器（可选）
c = 2048
```

小节名（`[my-gguf-model]`）会成为该模型在 API 中的 ID——随便命名。想加多少个模型就加多少个小节；`--models-max N` 限制同时可加载的数量。

`models.ini` 仍然是模型注册表和运行配置的来源；`/v1/models` 只是只读的发现接口，返回已经注册的模型、tags 和加载状态，不能替代 INI 或创建模型。INI 还负责模型 ID、`.gguf` / `.rkllm` 路径、`backend`、`mmproj`、上下文长度、加载策略、模板和推理选项。GGUF 与 RKLLM 都在各自小节填写 `tags`，服务端会统一通过 `/v1/models` 暴露。

**INI 键参考：**

| 键 | 含义 |
|---|---|
| `model` | 模型文件路径（`.rkllm` 或 `.gguf`） |
| `backend = rkllm` | 通过 `librkllmrt.so` 进程内加载。**省略**则走 GGUF / 子进程路径。 |
| `mmproj` | RKLLM 后端：**`.rknn` 视觉编码器**路径；GGUF 后端：标准 `mmproj-*.gguf` 投影器。可选。 |
| `tags` | GGUF/RKLLM 通用的可选 WebUI/API 标签，逗号分隔，例如 `0.8B,gguf` 或 `0.8B,rkllm,rknn`；服务从 INI 读取后统一显示在 `/v1/models`。 |
| `c` | 最大上下文长度（RKLLM 默认 2048） |
| `load-on-startup` | `true` 时在服务启动时**预加载**该模型 |
| `jinja`、`reasoning` | 模板 / 推理格式开关 |

**Router CLI 标志**（`common/common.h`、`common/arg.cpp`）：

| 标志 | 环境变量 | 默认 | 含义 |
|---|---|---|---|
| `--models-preset PATH` | `LLAMA_ARG_MODELS_PRESET` | 禁用 | 模型预设 INI 文件 |
| `--models-dir PATH` | `LLAMA_ARG_MODELS_DIR` | 禁用 | 扫描 `.gguf` 模型的目录 |
| `--models-max N` | `LLAMA_ARG_MODELS_MAX` | `4`（`0` = 无限） | 同时可加载的最大模型数 |
| `--default-model NAME` | `LLAMA_ARG_DEFAULT_MODEL` | 无 | 请求未指定 `model:` 时使用的模型 |
| `--models-autoload` / `--no-models-autoload` | `LLAMA_ARG_MODELS_AUTOLOAD` | 启用 | 请求命名了未加载模型时是否按需自动加载 |

注意三种"加载"机制的区别：

- **`load-on-startup`**（INI）——服务启动时预加载模型。
- **`--no-models-autoload`**（CLI）——阻止*按需*加载；请求未加载模型会报错而非触发加载。（它**不影响** `load-on-startup`。）
- **`--default-model`**（CLI）——请求未指定模型时回退到哪个模型。

### 3. 启动服务（Router 模式）

配置好 `models.ini` 后，**不带 `-m`** 启动 `llama-server` 并指向预设文件：

```sh
./build/bin/llama-server \
    --models-preset deploy/models.ini \
    --default-model my-rkllm-model \
    --models-max 1 \
    --host 0.0.0.0 --port 8080 -c 2048 --reasoning off
```

Web UI 随后可在 `http://<设备IP>:8080` 访问。模型通过 `/models/load` API、Web UI 或 `load-on-startup = true` 加载。设置了 `--default-model` 后，未带 `model` 字段的聊天请求会回退到该模型。

**Router API 速查**（注意路径——除注明外都不在 `/v1/` 下）：

| 方法与路径 | Body / query | 作用 |
|---|---|---|
| `GET /v1/models` | — | 列出所有模型及加载状态 |
| `POST /models/load` | `{"model":"<id>"}` | 加载（或排队）模型 |
| `POST /models/unload` | `{"model":"<id>"}` | 卸载模型 |
| `POST /v1/chat/completions` | OpenAI body（含 `model`） | 聊天补全 |

### 4. 多模态（视觉）输入

按 OpenAI vision API 风格传入图像——`content` 数组中包含 `type: "image_url"` 条目。支持的图像来源：

- 绝对路径：`/abs/path/image.jpg`
- `file:///abs/path/image.jpg`
- 数据 URI：`data:image/jpeg;base64,<base64 数据>`

若提示词中缺少 `<image>` 标签，会在图像嵌入之前自动注入一个 `<image>` 标签。

### 单模型模式（不用预设）

快速测试时可跳过预设，直接传入单个模型：

```sh
./build/bin/llama-server -m models/my-model.gguf --host 0.0.0.0 --port 8080 -c 2048
```

（进程内 RKLLM 通路仅通过 router + `backend = rkllm` 预设触发——`.rkllm` 文件没有单模型 `-m` 入口。）

---

## 部署

若要把 `llama-server` 作为开机自启的后台服务运行，[`deploy/`](./deploy/) 下提供了 systemd 单元和安装脚本。

### 1. 编辑 systemd 单元

打开 [`deploy/systemd/llama-server.service`](./deploy/systemd/llama-server.service)，把占位符替换为你的实际配置：

```ini
[Service]
User=<your-user>                      # 拥有仓库检出权限的账户
Group=<your-user>
WorkingDirectory=/opt/rk-llama.cpp    # 项目的绝对路径
Environment=LD_LIBRARY_PATH=/opt/rk-llama.cpp/ggml/src/ggml-rknpu2/libs
ExecStart=/opt/rk-llama.cpp/build/bin/llama-server \
    --models-preset /opt/rk-llama.cpp/deploy/models.ini \
    --default-model <model-id> \
    --models-max 1 \
    --host 0.0.0.0 --port 8080 -c 2048 --reasoning off
LimitNOFILE=65536
```

要点：

- `User=` / `Group=` —— 拥有仓库检出权限、且对 `/userdata/models/`（或你放置模型的位置）有读权限的账户。
- `WorkingDirectory` + `ExecStart` 中的两个绝对路径 —— 指向你的仓库检出。
- `LD_LIBRARY_PATH` —— 必须包含 `libs/` 目录，加载器才能找到 `librknnrt.so` / `librkllmrt.so`。单元同时设置了 `LimitNOFILE=65536`，因此服务侧不需要手动 `ulimit`。
- `--models-preset` —— 指向你编辑好的 `models.ini`（先从 `deploy/models.ini.example` 复制一份；见"运行"章节）。
- `--default-model <model-id>` —— 可选；`models.ini` 中的小节名，用于请求未带 `model:` 时提供服务。若希望每个请求都显式指定模型，删掉此行。

### 2. 安装并启用

[`deploy/install.sh`](./deploy/install.sh) 会把单元复制到 `/etc/systemd/system/` 并启用开机自启：

```sh
# 编辑单元文件后：
sudo bash deploy/install.sh
sudo systemctl start llama-server
sudo journalctl -u llama-server -f   # 查看日志
```

API 随后可在 `http://<设备IP>:8080` 访问。

### 其它部署说明

- **不用 systemd** —— 开发时可直接启动：`cd /opt/rk-llama.cpp && ulimit -n 65536 && ./build/bin/llama-server --models-preset deploy/models.ini ...`。
- **模型放在独立分区** —— 在 `/userdata` 作为大容量 eMMC/NVMe 分区的板子上很常见。把 `models.ini` 路径（如 `/userdata/models/...`）指向它即可；服务账户需要对模型文件有读权限。

---

## 状态与 TODO

这是一个进行中的端侧 AI 集成，而非成品。正在进行的工作：

- [ ] **RKNN 视觉编码器 + GGUF LLM 主干的混合** —— `tools/mtmd/rknn_encoder` 路径已存在，但需要集成测试，让 RKNN 编码器能够喂给 `.gguf` 主干（目前编码器仅接入了 RKLLM 主干）。
- [ ] **更广泛的模型支持** —— 在现有的 Qwen / Gemma 组合之外，测试更多 `.rkllm` LLM 与 `.rknn` 视觉编码器组合。
- [x] **驱动 / SDK 版本兼容性** —— 已验证 RKNN 驱动 `0.9.6+`/`0.9.8`、RKLLM 运行时 `1.2.3` 与 `1.3.0`（见分支 `runtime-1.3.0`）。更老/更新的版本仍需补充测试。
- [ ] **RK3576 配置** —— 补全 `ggml/src/ggml-rknpu2/` 中占位的设备配置。
- [x] **RKLLM 响应指标** —— RKLLM 通路现在提供 OpenAI 兼容的 `usage`（输入/输出/总 token 数）和 `timings`（延迟、tokens/s）；流式期间输出实时估计，结束时使用 SDK 的 `RKLLMPerfStat` 精确值。1.2.3 分支的请求级 `max_tokens` 仍受初始化上限限制。
- [ ] **工具 / 函数调用的健壮性**，以及 RKLLM 后端的流式输出边界情况。

---

## 致谢

本项目站在巨人的肩膀上：

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** —— The ggml authors，本项目所依赖的推理引擎。
- **[rk-llama.cpp](https://github.com/invisiofficial/rk-llama.cpp)** —— 本项目衍生于其 Rockchip NPU GGML 后端分支。
- **瑞芯微（Rockchip）** —— RKNN & RKLLM SDK（运行时库与头文件位于 `ggml/src/ggml-rknpu2/libs/`）。

第三方署名与许可说明见 [NOTICE.md](./NOTICE.md)。

---

## 许可证

MIT 协议——见 [LICENSE](./LICENSE)。

内置的瑞芯微 RKNN / RKLLM SDK 文件（头文件 `rknn_api.h`、`rknn_custom_op.h`、`rknn_matmul_api.h`、`rkllm.h`，以及 `librknnrt.so` / `librkllmrt.so` 二进制文件）**归瑞芯微所有**，受其自身条款约束，**不**适用本仓库的 MIT 许可证。详见 [NOTICE.md](./NOTICE.md)。
