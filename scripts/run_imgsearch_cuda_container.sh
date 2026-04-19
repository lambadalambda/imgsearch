#!/usr/bin/env bash
set -euo pipefail

data_dir="${IMGSEARCH_DATA_DIR:-/data}"

mkdir -p "${data_dir}" /models

exec /opt/imgsearch/bin/imgsearch \
  -data-dir "${data_dir}" \
  -addr "${IMGSEARCH_ADDR:-0.0.0.0:8080}" \
  -mode "${IMGSEARCH_MODE:-all}" \
  -vector-backend sqlite-vector \
  -sqlite-vector-path "${SQLITE_VECTOR_PATH:-/opt/imgsearch/tools/sqlite-vector/vector}" \
  -enable-annotations="${IMGSEARCH_ENABLE_ANNOTATIONS:-true}" \
  -llama-native-model-path "${IMGSEARCH_LLAMA_MODEL_PATH:-/models/Qwen/Qwen3-VL-Embedding-8B-Q4_K_M.gguf}" \
  -llama-native-mmproj-path "${IMGSEARCH_LLAMA_MMPROJ_PATH:-/models/Qwen/mmproj-Qwen3-VL-Embedding-8B-f16.gguf}" \
  -llama-native-annotator-model-path "${IMGSEARCH_ANNOTATOR_MODEL_PATH:-/models/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf}" \
  -llama-native-annotator-mmproj-path "${IMGSEARCH_ANNOTATOR_MMPROJ_PATH:-/models/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive/mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf}" \
  "$@"
