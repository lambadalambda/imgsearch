#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

args_file="$tmp_dir/args.txt"
mock_bin="$tmp_dir/imgsearch"

cat >"$mock_bin" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

: "${IMGSEARCH_TEST_ARGS_FILE:?missing IMGSEARCH_TEST_ARGS_FILE}"
printf '%s\n' "$@" >"$IMGSEARCH_TEST_ARGS_FILE"
EOF
chmod +x "$mock_bin"

extract_flag_value() {
  local flag="$1"
  local file="$2"
  local previous=""
  while IFS= read -r arg; do
    if [[ "$previous" == "$flag" ]]; then
      printf '%s' "$arg"
      return 0
    fi
    previous="$arg"
  done <"$file"
  return 1
}

run_entrypoint() {
  : >"$args_file"
  IMGSEARCH_TEST_ARGS_FILE="$args_file" \
    IMGSEARCH_BIN="$mock_bin" \
    IMGSEARCH_DATA_DIR="$tmp_dir/data" \
    IMGSEARCH_MODELS_DIR="$tmp_dir/models" \
    "$repo_root/scripts/run_imgsearch_cuda_container.sh" >/dev/null
  if [[ ! -s "$args_file" ]]; then
    echo "expected mock binary to be invoked" >&2
    exit 1
  fi
}

run_entrypoint
default_addr="$(extract_flag_value "-addr" "$args_file")"
if [[ "$default_addr" != "127.0.0.1:8080" ]]; then
  echo "expected default addr 127.0.0.1:8080, got '$default_addr'" >&2
  exit 1
fi
default_model="$(extract_flag_value "-llama-native-model-path" "$args_file")"
if [[ "$default_model" != "$tmp_dir/models/VesNFF/Qwen3-VL-Embedding-2B-GGUF/Qwen3-VL-Embedding-2B-Q6_K.gguf" ]]; then
  echo "expected default 2B model path, got '$default_model'" >&2
  exit 1
fi
default_mmproj="$(extract_flag_value "-llama-native-mmproj-path" "$args_file")"
if [[ "$default_mmproj" != "$tmp_dir/models/VesNFF/Qwen3-VL-Embedding-2B-GGUF/mmproj-Qwen3-VL-Embedding-2B-f16.gguf" ]]; then
  echo "expected default 2B mmproj path, got '$default_mmproj'" >&2
  exit 1
fi
default_dims="$(extract_flag_value "-llama-native-dimensions" "$args_file")"
if [[ "$default_dims" != "2048" ]]; then
  echo "expected default dimensions 2048, got '$default_dims'" >&2
  exit 1
fi

: >"$args_file"
IMGSEARCH_TEST_ARGS_FILE="$args_file" \
  IMGSEARCH_BIN="$mock_bin" \
  IMGSEARCH_DATA_DIR="$tmp_dir/data" \
  IMGSEARCH_MODELS_DIR="$tmp_dir/models" \
  IMGSEARCH_ADDR="0.0.0.0:8080" \
  "$repo_root/scripts/run_imgsearch_cuda_container.sh" >/dev/null
if [[ ! -s "$args_file" ]]; then
  echo "expected mock binary to be invoked" >&2
  exit 1
fi

public_addr="$(extract_flag_value "-addr" "$args_file")"
if [[ "$public_addr" != "0.0.0.0:8080" ]]; then
  echo "expected explicit addr override, got '$public_addr'" >&2
  exit 1
fi

echo "ok"
