#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

fixture_repo="$tmp_dir/repo"
fake_bin="$tmp_dir/bin"
mkdir -p "$fixture_repo/scripts" "$fixture_repo/deps/llama.cpp" "$fake_bin"
cp "$repo_root/scripts/ensure_llama_cpp_native_build.sh" "$fixture_repo/scripts/ensure_llama_cpp_native_build.sh"
chmod +x "$fixture_repo/scripts/ensure_llama_cpp_native_build.sh"

case "$(uname -s)" in
  Darwin)
    llama_lib="libllama.dylib"
    common_lib="libllama-common.dylib"
    ;;
  Linux)
    llama_lib="libllama.so"
    common_lib="libllama-common.so"
    ;;
  *)
    echo "unsupported host platform: $(uname -s)" >&2
    exit 1
    ;;
esac

cmake_log="$tmp_dir/cmake.log"
cat >"$fake_bin/cmake" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

: "${IMGSEARCH_TEST_CMAKE_LOG:?missing IMGSEARCH_TEST_CMAKE_LOG}"
: "${IMGSEARCH_TEST_BUILD_DIR:?missing IMGSEARCH_TEST_BUILD_DIR}"
: "${IMGSEARCH_TEST_LLAMA_LIB:?missing IMGSEARCH_TEST_LLAMA_LIB}"
: "${IMGSEARCH_TEST_COMMON_LIB:?missing IMGSEARCH_TEST_COMMON_LIB}"

printf '%s\n' "$*" >>"$IMGSEARCH_TEST_CMAKE_LOG"
mkdir -p "$IMGSEARCH_TEST_BUILD_DIR/bin"
touch "$IMGSEARCH_TEST_BUILD_DIR/bin/$IMGSEARCH_TEST_LLAMA_LIB"
touch "$IMGSEARCH_TEST_BUILD_DIR/bin/$IMGSEARCH_TEST_COMMON_LIB"
EOF
chmod +x "$fake_bin/cmake"

run_build_helper() {
  env \
    IMGSEARCH_TEST_CMAKE_LOG="$cmake_log" \
    IMGSEARCH_TEST_BUILD_DIR="$fixture_repo/deps/llama.cpp/build" \
    IMGSEARCH_TEST_LLAMA_LIB="$llama_lib" \
    IMGSEARCH_TEST_COMMON_LIB="$common_lib" \
    PATH="$fake_bin:$PATH" \
    "$@" \
    "$fixture_repo/scripts/ensure_llama_cpp_native_build.sh" >/dev/null
}

build_dir="$fixture_repo/deps/llama.cpp/build"
mkdir -p "$build_dir/bin" "$build_dir/common"
touch "$build_dir/bin/$llama_lib"
touch "$build_dir/common/libcommon.a"
touch "$build_dir/stale-marker"

run_build_helper IMGSEARCH_LLAMA_BUILD_JOBS=2

if [[ -e "$build_dir/stale-marker" ]]; then
  echo "expected stale build dir to be removed when libllama-common is missing" >&2
  exit 1
fi
if [[ ! -f "$build_dir/bin/$common_lib" ]]; then
  echo "expected fake cmake to create $common_lib" >&2
  exit 1
fi
if [[ ! -s "$cmake_log" ]]; then
  echo "expected cmake to run after stale build cleanup" >&2
  exit 1
fi
cmake_log_contents="$(<"$cmake_log")"
if [[ "$cmake_log_contents" != *"--target llama mtmd llama-common"* ]]; then
  echo "expected build helper to build only required native library targets" >&2
  exit 1
fi
if [[ "$cmake_log_contents" == *"llama-server"* ]]; then
  echo "expected build helper not to build the llama.cpp server target" >&2
  exit 1
fi
if [[ "$cmake_log_contents" != *"-j 2"* ]]; then
  echo "expected build helper to honor IMGSEARCH_LLAMA_BUILD_JOBS" >&2
  exit 1
fi

: >"$cmake_log"
run_build_helper
if [[ -s "$cmake_log" ]]; then
  echo "expected existing libllama-common build to be reused" >&2
  exit 1
fi

echo "ok"
