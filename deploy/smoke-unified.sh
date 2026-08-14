#!/usr/bin/env bash

# Smoke test for the experimental in-process model manager.
# Required: curl, python3, a router started with LLAMA_SERVER_IN_PROCESS_GGUF=1.
# Required environment: GGUF_MODEL and RKLLM_MODEL.
# Optional: BASE_URL, GGUF_TAG, RKLLM_TAG.

set -eu

BASE_URL="${BASE_URL:-http://127.0.0.1:8080}"
GGUF_MODEL="${GGUF_MODEL:?set GGUF_MODEL to the GGUF model id}"
RKLLM_MODEL="${RKLLM_MODEL:?set RKLLM_MODEL to the RKLLM model id}"

request() {
    local method="$1"
    local path="$2"
    local body="${3:-}"
    if [ -n "$body" ]; then
        curl --silent --show-error --write-out $'\n%{http_code}' \
            -X "$method" "$BASE_URL$path" \
            -H 'Content-Type: application/json' \
            --data-binary "$body"
    else
        curl --silent --show-error --write-out $'\n%{http_code}' \
            -X "$method" "$BASE_URL$path"
    fi
}

split_response() {
    RESPONSE_BODY="${1%$'\n'*}"
    RESPONSE_STATUS="${1##*$'\n'}"
}

assert_status() {
    local response="$1"
    local expected="$2"
    split_response "$response"
    if [ "$RESPONSE_STATUS" != "$expected" ]; then
        printf 'unexpected HTTP status: got %s, expected %s\n%s\n' \
            "$RESPONSE_STATUS" "$expected" "$RESPONSE_BODY" >&2
        exit 1
    fi
}

assert_json_models() {
    python3 - "$1" "$2" "$3" "${4:-}" "${5:-}" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
expected = {
    sys.argv[2]: ("llama", sys.argv[4]),
    sys.argv[3]: ("rkllm", sys.argv[5]),
}
models = {item["id"]: item for item in payload.get("data", [])}
for model_id, (backend, expected_tag) in expected.items():
    if model_id not in models:
        raise SystemExit(f"model {model_id!r} missing from /v1/models")
    item = models[model_id]
    if item.get("backend") != backend:
        raise SystemExit(f"{model_id}: backend={item.get('backend')!r}, expected {backend!r}")
    if expected_tag and expected_tag not in item.get("tags", []):
        raise SystemExit(f"{model_id}: tag {expected_tag!r} missing from {item.get('tags')!r}")
PY
}

assert_stream_metrics() {
    python3 - "$1" <<'PY'
import json
import sys

found_usage = False
found_timings = False
for line in sys.argv[1].splitlines():
    if not line.startswith("data: ") or line == "data: [DONE]":
        continue
    item = json.loads(line[6:])
    if "usage" in item:
        usage = item["usage"]
        if usage.get("total_tokens", 0) <= 0:
            raise SystemExit("usage.total_tokens is not positive")
        found_usage = True
    if "timings" in item:
        timings = item["timings"]
        if "predicted_n" not in timings or "predicted_ms" not in timings:
            raise SystemExit("timings is missing predicted token fields")
        found_timings = True
if not found_usage:
    raise SystemExit("stream did not contain a usage chunk")
if not found_timings:
    raise SystemExit("stream did not contain a timings chunk")
PY
}

assert_json_completion() {
    python3 - "$1" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
choices = payload.get("choices", [])
if not choices:
    raise SystemExit("completion response has no choices")
message = choices[0].get("message", {})
if not message.get("content"):
    raise SystemExit("completion response has empty message content")
if payload.get("usage", {}).get("total_tokens", 0) <= 0:
    raise SystemExit("completion response has no token usage")
PY
}

load_model() {
    local model="$1"
    local response
    response="$(request POST /models/load "$(printf '{"model":"%s"}' "$model")")"
    split_response "$response"
    case "$RESPONSE_STATUS" in
        2??) ;;
        400)
            case "$RESPONSE_BODY" in
                *"already running"*) ;;
                *) printf 'failed to load %s:\n%s\n' "$model" "$RESPONSE_BODY" >&2; exit 1 ;;
            esac
            ;;
        *) printf 'failed to load %s:\n%s\n' "$model" "$RESPONSE_BODY" >&2; exit 1 ;;
    esac
}

unload_model() {
    local model="$1"
    local response
    response="$(request POST /models/unload "$(printf '{"model":"%s"}' "$model")")"
    split_response "$response"
    case "$RESPONSE_STATUS" in
        2??) ;;
        400)
            case "$RESPONSE_BODY" in
                *"not running"*) ;;
                *) printf 'failed to unload %s:\n%s\n' "$model" "$RESPONSE_BODY" >&2; exit 1 ;;
            esac
            ;;
        *) printf 'failed to unload %s:\n%s\n' "$model" "$RESPONSE_BODY" >&2; exit 1 ;;
    esac
}

response="$(request GET /v1/models)"
assert_status "$response" 200
split_response "$response"
assert_json_models "$RESPONSE_BODY" "$GGUF_MODEL" "$RKLLM_MODEL" \
    "${GGUF_TAG:-}" "${RKLLM_TAG:-}"
printf 'PASS /v1/models backend and tags\n'

# Loading RKLLM after GGUF exercises the supported direction and the NPU
# residency hand-off. The reverse direction is checked below in this process.
load_model "$GGUF_MODEL"

chat_body="$(python3 - "$GGUF_MODEL" <<'PY'
import json
import sys
print(json.dumps({
    "model": sys.argv[1],
    "messages": [{"role": "user", "content": "Reply with one short word."}],
    "stream": False,
}))
PY
)"
response="$(request POST /v1/chat/completions "$chat_body")"
assert_status "$response" 200
split_response "$response"
assert_json_completion "$RESPONSE_BODY"
printf 'PASS GGUF inference and usage\n'

load_model "$RKLLM_MODEL"

chat_body="$(python3 - "$RKLLM_MODEL" <<'PY'
import json
import sys
print(json.dumps({
    "model": sys.argv[1],
    "messages": [{"role": "user", "content": "Reply with one short word."}],
    "stream": True,
    "stream_options": {"include_usage": True},
}))
PY
)"
response="$(request POST /v1/chat/completions "$chat_body")"
assert_status "$response" 200
split_response "$response"
assert_stream_metrics "$RESPONSE_BODY"
printf 'PASS RKLLM streaming usage and timings\n'

unload_model "$RKLLM_MODEL"
response="$(request POST /models/load "$(printf '{"model":"%s"}' "$GGUF_MODEL")")"
split_response "$response"
if [ "$RESPONSE_STATUS" != 500 ] ||
   ! printf '%s' "$RESPONSE_BODY" | grep -q 'restart llama-server to reset the RKNN runtime'; then
    printf 'expected RKLLM-to-GGUF guard, got HTTP %s:\n%s\n' \
        "$RESPONSE_STATUS" "$RESPONSE_BODY" >&2
    exit 1
fi
printf 'PASS RKLLM-to-GGUF guard\n'

response="$(request GET /v1/models)"
assert_status "$response" 200
printf 'PASS router remains alive after guarded switch\n'
