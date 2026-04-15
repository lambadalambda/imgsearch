#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
addr="${IMGSEARCH_SMOKE_ADDR:-127.0.0.1:18081}"
data_dir="$(mktemp -d)"
bin="${TMPDIR:-/tmp}/imgsearch-smoke-serve"
pid=""

cleanup() {
  if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
    wait "${pid}" 2>/dev/null || true
  fi
  rm -rf "${data_dir}"
}
trap cleanup EXIT

"${repo_root}/scripts/ensure_sqlite_vector_ready.sh"
"${repo_root}/scripts/ensure_llama_cpp_native_build.sh"

go build -o "${bin}" "${repo_root}/cmd/imgsearch"

SQLITE_VECTOR_PATH="${repo_root}/tools/sqlite-vector/vector" \
  "${bin}" \
  -mode api \
  -enable-annotations=false \
  -addr "${addr}" \
  -data-dir "${data_dir}" \
  -vector-backend sqlite-vector \
  -llama-native-model-path "${LLAMA_NATIVE_MODEL_PATH:-${repo_root}/models/Qwen/Qwen3-VL-Embedding-8B-Q4_K_M.gguf}" \
  -llama-native-mmproj-path "${LLAMA_NATIVE_MMPROJ_PATH:-${repo_root}/models/Qwen/mmproj-Qwen3-VL-Embedding-8B-f16.gguf}" \
  -llama-native-dimensions "${LLAMA_NATIVE_DIMS:-4096}" \
  -llama-native-gpu-layers "${LLAMA_NATIVE_GPU_LAYERS:-99}" \
  -llama-native-use-gpu="${LLAMA_NATIVE_USE_GPU:-true}" \
  -llama-native-context-size "${LLAMA_NATIVE_CONTEXT_SIZE:-8192}" \
  -llama-native-batch-size "${LLAMA_NATIVE_BATCH_SIZE:-512}" \
  -llama-native-threads "${LLAMA_NATIVE_THREADS:-0}" \
  -llama-native-image-max-side "${LLAMA_NATIVE_IMAGE_MAX_SIDE:-512}" \
  -llama-native-image-max-tokens "${LLAMA_NATIVE_IMAGE_MAX_TOKENS:-0}" \
  >/tmp/imgsearch-smoke-serve.log 2>&1 &
pid="$!"

for _ in $(seq 1 60); do
  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    cat /tmp/imgsearch-smoke-serve.log >&2 || true
    echo "imgsearch exited before becoming healthy" >&2
    exit 1
  fi
  if curl -fsS "http://${addr}/healthz" >/dev/null 2>&1; then
    exit 0
  fi
  sleep 1
done

cat /tmp/imgsearch-smoke-serve.log >&2 || true
echo "imgsearch did not become healthy in time" >&2
exit 1
