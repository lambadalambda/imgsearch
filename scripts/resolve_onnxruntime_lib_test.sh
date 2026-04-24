#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

mkdir -p "$tmp_dir/test_data"
touch "$tmp_dir/test_data/onnxruntime_arm64.dylib"
touch "$tmp_dir/test_data/onnxruntime_arm64.so"

assert_resolves() {
  local os_name="$1"
  local arch="$2"
  local want="$3"
  local got
  got="$(ONNXRUNTIME_GO_MODULE_DIR="$tmp_dir" IMGSEARCH_TEST_UNAME_S="$os_name" IMGSEARCH_TEST_UNAME_M="$arch" "$repo_root/scripts/resolve_onnxruntime_lib.sh")"
  if [[ "$got" != "$tmp_dir/test_data/$want" ]]; then
    echo "expected $os_name/$arch to resolve $want, got $got" >&2
    exit 1
  fi
}

assert_resolves Darwin arm64 onnxruntime_arm64.dylib
assert_resolves Darwin aarch64 onnxruntime_arm64.dylib
assert_resolves Linux arm64 onnxruntime_arm64.so
assert_resolves Linux aarch64 onnxruntime_arm64.so

if ONNXRUNTIME_GO_MODULE_DIR="$tmp_dir" IMGSEARCH_TEST_UNAME_S=Linux IMGSEARCH_TEST_UNAME_M=x86_64 "$repo_root/scripts/resolve_onnxruntime_lib.sh" >/dev/null 2>&1; then
  echo "expected Linux x86_64 to fail until onnxruntime_go ships a matching test library" >&2
  exit 1
fi

if ONNXRUNTIME_GO_MODULE_DIR="$tmp_dir" IMGSEARCH_TEST_UNAME_S=Darwin IMGSEARCH_TEST_UNAME_M=x86_64 "$repo_root/scripts/resolve_onnxruntime_lib.sh" >/dev/null 2>&1; then
  echo "expected Darwin x86_64 to fail until onnxruntime_go ships a matching test library" >&2
  exit 1
fi

echo "ok"
