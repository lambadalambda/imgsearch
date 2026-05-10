#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mise_toml="$(<"${repo_root}/mise.toml")"

assert_contains() {
  local needle="$1"
  if [[ "${mise_toml}" != *"${needle}"* ]]; then
    echo "expected mise.toml to contain: ${needle}" >&2
    exit 1
  fi
}

assert_contains '[tasks.serve]'
assert_contains 'mise watch serve:run --restart'
assert_contains '--no-vcs-ignore --watch cmd --watch internal'
assert_contains 'Qwen3-VL-Embedding-2B-Q6_K.gguf'
assert_contains 'LLAMA_NATIVE_DIMS:-2048'
assert_contains '[tasks."serve:8b"]'
assert_contains 'mise watch serve:8b:run --restart'
assert_contains '[tasks."serve:8b:run"]'
assert_contains 'Qwen3-VL-Embedding-8B-Q4_K_M.gguf'
assert_contains 'LLAMA_NATIVE_DIMS:-4096'
assert_contains '[tasks."serve:8b:annotator-26b"]'
assert_contains 'mise watch serve:8b:annotator-26b:run --restart'
assert_contains '[tasks."serve:8b:annotator-26b:run"]'

echo "ok"
