#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
resolver="${repo_root}/scripts/resolve_onnxruntime_lib.sh"
tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

version="9.9.9"
mkdir -p "$tmp_dir/cache/$version/linux-x64/lib" "$tmp_dir/cache/$version/linux-aarch64/lib" "$tmp_dir/cache/$version/osx-arm64/lib"
touch "$tmp_dir/cache/$version/linux-x64/lib/libonnxruntime.so.$version"
touch "$tmp_dir/cache/$version/linux-aarch64/lib/libonnxruntime.so.$version"
touch "$tmp_dir/cache/$version/osx-arm64/lib/libonnxruntime.$version.dylib"

resolve() {
  IMGSEARCH_ONNXRUNTIME_CACHE_DIR="$tmp_dir/cache" IMGSEARCH_ONNXRUNTIME_VERSION="$version" IMGSEARCH_ONNXRUNTIME_DOWNLOAD=0 \
    IMGSEARCH_TEST_UNAME_S="$1" IMGSEARCH_TEST_UNAME_M="$2" "$resolver"
}

assert_resolves() {
  local os_name="$1"
  local arch="$2"
  local want="$3"
  local got
  got="$(resolve "$os_name" "$arch")"
  if [[ "$got" != "$tmp_dir/cache/$version/$want" ]]; then
    echo "expected $os_name/$arch to resolve $want, got $got" >&2
    exit 1
  fi
}

assert_resolves Linux x86_64 "linux-x64/lib/libonnxruntime.so.$version"
assert_resolves Linux aarch64 "linux-aarch64/lib/libonnxruntime.so.$version"
assert_resolves Linux arm64 "linux-aarch64/lib/libonnxruntime.so.$version"
assert_resolves Darwin arm64 "osx-arm64/lib/libonnxruntime.$version.dylib"

# Explicit override wins and must exist.
touch "$tmp_dir/custom.so"
got="$(IMGSEARCH_ONNXRUNTIME_LIB="$tmp_dir/custom.so" "$resolver")"
if [[ "$got" != "$tmp_dir/custom.so" ]]; then
  echo "expected IMGSEARCH_ONNXRUNTIME_LIB override, got $got" >&2
  exit 1
fi
if IMGSEARCH_ONNXRUNTIME_LIB="$tmp_dir/missing.so" "$resolver" >/dev/null 2>&1; then
  echo "expected a missing IMGSEARCH_ONNXRUNTIME_LIB to fail" >&2
  exit 1
fi

# Platforms without an official build fail with an actionable message.
if msg="$(resolve Darwin x86_64 2>&1)"; then
  echo "expected Darwin x86_64 to fail without an official build" >&2
  exit 1
fi
if [[ "$msg" != *"IMGSEARCH_ONNXRUNTIME_LIB"* ]]; then
  echo "expected the unsupported-platform message to name IMGSEARCH_ONNXRUNTIME_LIB, got: $msg" >&2
  exit 1
fi

# A cache miss with downloads disabled explains what is missing.
if msg="$(IMGSEARCH_ONNXRUNTIME_CACHE_DIR="$tmp_dir/empty" IMGSEARCH_ONNXRUNTIME_VERSION="$version" IMGSEARCH_ONNXRUNTIME_DOWNLOAD=0 IMGSEARCH_TEST_UNAME_S=Linux IMGSEARCH_TEST_UNAME_M=x86_64 "$resolver" 2>&1)"; then
  echo "expected a cache miss with downloads disabled to fail" >&2
  exit 1
fi
if [[ "$msg" != *"not cached"* ]]; then
  echo "expected the cache-miss message, got: $msg" >&2
  exit 1
fi

echo "ok"
