# NOTICE

This project incorporates work from several sources. This file acknowledges them
and clarifies which license terms apply to which components.

## rk-llama.cpp (this fork)

The additions in this repository — the RKLLM in-process backend
(`tools/server/rkllm-instance.*`), the multimodal RKNN vision encoder path, and
the router integration — are licensed under the MIT License, see [LICENSE](./LICENSE).

Copyright (c) 2026 rk-llama.cpp contributors.

## llama.cpp

The core inference engine is [llama.cpp](https://github.com/ggml-org/llama.cpp)
by **The ggml authors**, licensed under the MIT License.

Copyright (c) 2023-2026 The ggml authors.

## rk-llama.cpp (upstream fork)

The Rockchip NPU GGML backend (`ggml/src/ggml-rknpu2/`) originates from
[invisiofficial/rk-llama.cpp](https://github.com/invisiofficial/rk-llama.cpp),
a fork of llama.cpp, licensed under the MIT License.

## Rockchip RKNN & RKLLM SDK (proprietary — NOT MIT)

The following vendored files under `ggml/src/ggml-rknpu2/libs/` are **proprietary
to Rockchip Corporation** and are governed by Rockchip's own license terms,
**not** the MIT License of this repository:

- `libs/include/rknn_api.h`
- `libs/include/rknn_custom_op.h`
- `libs/include/rknn_matmul_api.h`
- `libs/include/rkllm.h`
- `libs/librknnrt.so`
- `libs/librkllmrt.so`

The Rockchip headers carry the following notice (quoted verbatim from
`rknn_api.h`):

> Copyright (c) 2017 - 2022 by Rockchip Corp. All rights reserved.
>
> The material in this file is confidential and contains trade secrets of
> Rockchip Corporation. This is proprietary information owned by Rockchip
> Corporation. No part of this work may be disclosed, reproduced, copied,
> transmitted, or used in any way for any purpose, without the express written
> permission of Rockchip Corporation.

> **Note:** These SDK artifacts are redistributed here for convenience. Anyone
> using, modifying, or redistributing this repository is responsible for
> ensuring they comply with Rockchip's SDK terms and for obtaining any required
> permissions from Rockchip. The MIT license of the surrounding code does not
> extend to these proprietary files.
