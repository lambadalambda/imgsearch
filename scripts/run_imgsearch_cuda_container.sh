#!/usr/bin/env bash
set -euo pipefail

data_dir="${IMGSEARCH_DATA_DIR:-/data}"
models_dir="${IMGSEARCH_MODELS_DIR:-/models}"
imgsearch_bin="${IMGSEARCH_BIN:-/opt/imgsearch/bin/imgsearch}"
addr="${IMGSEARCH_ADDR:-127.0.0.1:8080}"

mkdir -p "${data_dir}" "${models_dir}"

exec "${imgsearch_bin}" \
  -data-dir "${data_dir}" \
  -addr "${addr}" \
  -mode "${IMGSEARCH_MODE:-all}" \
  -vector-backend sqlite-vector \
  -sqlite-vector-path "${SQLITE_VECTOR_PATH:-/opt/imgsearch/tools/sqlite-vector/vector}" \
  -enable-annotations="${IMGSEARCH_ENABLE_ANNOTATIONS:-true}" \
  -llama-native-model-path "${IMGSEARCH_LLAMA_MODEL_PATH:-${models_dir}/Qwen/Qwen3-VL-Embedding-8B-Q4_K_M.gguf}" \
  -llama-native-mmproj-path "${IMGSEARCH_LLAMA_MMPROJ_PATH:-${models_dir}/Qwen/mmproj-Qwen3-VL-Embedding-8B-f16.gguf}" \
  -llama-native-annotator-model-path "${IMGSEARCH_ANNOTATOR_MODEL_PATH:-${models_dir}/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf}" \
  -llama-native-annotator-mmproj-path "${IMGSEARCH_ANNOTATOR_MMPROJ_PATH:-${models_dir}/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive/mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf}" \
  "$@"
