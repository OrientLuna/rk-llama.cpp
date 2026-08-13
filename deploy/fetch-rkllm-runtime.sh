#!/bin/bash
# fetch-rkllm-runtime.sh — download/swap the Rockchip RKLLM runtime (librkllmrt.so)
#
# The .rkllm model format is version-locked to the rkllm-toolkit/runtime that
# produced it: a model converted with toolkit 1.3.0 needs librkllmrt.so 1.3.0.
# This script fetches a specific runtime version from the airockchip/rknn-llm
# repo and installs it into ggml/src/ggml-rknpu2/libs/, backing up the current one.
#
# Usage:
#   ./deploy/fetch-rkllm-runtime.sh <version>   # e.g. 1.3.0, 1.2.3
#   ./deploy/fetch-rkllm-runtime.sh --current   # show the currently installed version
#   ./deploy/fetch-rkllm-runtime.sh --list      # list known available versions
#
# Env:
#   LIBRKLLMRT_ARCH  target arch (default: aarch64; also: armhf)
#   LIBRKLLMRT_MIRROR  override the download mirror (default: auto github/hf-mirror)
set -euo pipefail

LIBS_DIR="$(cd "$(dirname "$0")/.." && pwd)/ggml/src/ggml-rknpu2/libs"
SO_PATH="$LIBS_DIR/librkllmrt.so"
ARCH="${LIBRKLLMRT_ARCH:-aarch64}"
REPO="airockchip/rknn-llm"
# Path is stable from v1.1.4 onward
RUNTIME_PATH="rkllm-runtime/Linux/librkllm_api/${ARCH}/librkllmrt.so"

# Known release tags (version => tag)
declare -A KNOWN=(
  ["1.3.0"]="release-v1.3.0"
  ["1.2.3"]="release-v1.2.3"
  ["1.2.2"]="release-v1.2.2"
  ["1.2.1"]="release-v1.2.1"
  ["1.2.0"]="release-v1.2.0"
  ["1.1.4"]="release-v1.1.4"
)

current_version() {
  if [ ! -f "$SO_PATH" ]; then echo "(not installed)"; return; fi
  strings "$SO_PATH" | grep -oE "version: [0-9.]+" | head -1 | grep -oE "[0-9.]+" || echo "(unknown)"
}

cmd_current() {
  echo "librkllmrt.so: $(current_version)"
  echo "  arch: $ARCH"
  echo "  path: $SO_PATH"
}

cmd_list() {
  echo "Known RKLLM runtime versions (from $REPO):"
  for v in 1.3.0 1.2.3 1.2.2 1.2.1 1.2.0 1.1.4; do
    tag="${KNOWN[$v]}"
    mark=""
    [ "$(current_version)" = "$v" ] && mark="  <- current"
    printf "  %s  (tag: %s)%s\n" "$v" "$tag" "$mark"
  done
}

download() {
  local ver="$1"
  local tag="${KNOWN[$ver]:-}"
  if [ -z "$tag" ]; then
    # Allow arbitrary versions by constructing the tag name
    tag="release-v$ver"
    echo "(version '$ver' not in known list; trying tag '$tag')"
  fi

  local cur; cur="$(current_version)"
  if [ "$cur" = "$ver" ]; then
    echo "Version $ver is already installed. Nothing to do."
    exit 0
  fi

  # Build candidate URLs (github raw, then hf-mirror fallback for CN networks)
  local urls=(
    "https://github.com/${REPO}/raw/refs/tags/${tag}/${RUNTIME_PATH}"
    "https://hf-mirror.com/${REPO}/resolve/${tag}/${RUNTIME_PATH}"
  )
  if [ -n "${LIBRKLLMRT_MIRROR:-}" ]; then
    urls=("${LIBRKLLMRT_MIRROR}/${REPO}/resolve/${tag}/${RUNTIME_PATH}")
  fi

  local tmp; tmp="$(mktemp)"
  echo "Downloading librkllmrt.so $ver ($ARCH) ..."
  local ok=0
  for url in "${urls[@]}"; do
    echo "  trying: $url"
    if curl -fSL --connect-timeout 15 -o "$tmp" "$url" 2>/dev/null; then
      ok=1; echo "  downloaded from this mirror."; break
    fi
    echo "  failed."
  done
  if [ "$ok" -ne 1 ]; then
    echo "ERROR: could not download librkllmrt.so $ver from any mirror." >&2
    echo "       Check the version/tag, network, or set LIBRKLLMRT_MIRROR." >&2
    rm -f "$tmp"; exit 1
  fi

  # Verify it's an ELF and the version string matches
  if ! file "$tmp" | grep -q "ELF"; then
    echo "ERROR: downloaded file is not an ELF binary (got an HTML error page?)." >&2
    rm -f "$tmp"; exit 1
  fi
  local got; got="$(strings "$tmp" | grep -oE "version: [0-9.]+" | head -1 | grep -oE "[0-9.]+" || echo unknown)"
  if [ "$got" != "$ver" ]; then
    echo "WARNING: requested $ver but downloaded file reports version '$got'." >&2
    echo "         Aborting to avoid a version mismatch. Inspect $tmp manually." >&2
    exit 1
  fi

  # Backup current + install
  if [ -f "$SO_PATH" ] && [ "$cur" != "(unknown)" ]; then
    local bak="$LIBS_DIR/librkllmrt.so.${cur}.bak"
    cp "$SO_PATH" "$bak"
    echo "Backed up current ($cur) -> $(basename "$bak")"
  fi
  install -m 0644 "$tmp" "$SO_PATH"
  rm -f "$tmp"
  echo "Installed librkllmrt.so $ver -> $SO_PATH"
  echo ""
  echo "NOTE: if you switched to a different major version, rebuild:"
  echo "  cd build && cmake .. -DLLAMA_RKNPU2=ON && make -j\$(nproc) llama-server"
  echo "(This repo's runtime-1.3.0 branch has the matching header + code for 1.3.0.)"
}

usage() {
  sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
}

case "${1:-}" in
  --current) cmd_current ;;
  --list)    cmd_list ;;
  -h|--help) usage ;;
  "")        usage ;;
  *)         download "$1" ;;
esac
