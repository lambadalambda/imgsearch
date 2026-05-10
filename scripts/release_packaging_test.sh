#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
package_script="$(<"${repo_root}/scripts/package_release.sh")"
build_script="$(<"${repo_root}/scripts/ensure_llama_cpp_native_build.sh")"
release_workflow="$(<"${repo_root}/.github/workflows/rolling-release.yml")"

assert_contains() {
  local haystack="$1"
  local needle="$2"
  if [[ "${haystack}" != *"${needle}"* ]]; then
    echo "expected release packaging config to contain: ${needle}" >&2
    exit 1
  fi
}

assert_not_contains() {
  local haystack="$1"
  local needle="$2"
  if [[ "${haystack}" == *"${needle}"* ]]; then
    echo "expected release packaging config not to contain: ${needle}" >&2
    exit 1
  fi
}

assert_contains "${package_script}" 'build_atelier_frontend'
assert_contains "${package_script}" 'npm ci --no-audit --no-fund --silent'
assert_contains "${package_script}" 'npm run build'
assert_contains "${package_script}" 'The default 2B Qwen GGUF files'
assert_contains "${package_script}" 'Qwen3-VL-Embedding-8B-Q4_K_M.gguf'
assert_contains "${package_script}" '-llama-native-dimensions 4096'
assert_contains "${package_script}" 'SQLITE_VECTOR_PATH="$script_dir/tools/sqlite-vector/vector"'
assert_contains "${package_script}" 'exec "$script_dir/imgsearch" -vector-backend sqlite-vector "$@"'
assert_not_contains "${package_script}" 'The default 8B Qwen GGUF files'
assert_contains "${build_script}" 'IMGSEARCH_LLAMA_CMAKE_ARGS'
assert_contains "${release_workflow}" 'actions/setup-node@v4'
assert_contains "${release_workflow}" 'IMGSEARCH_LLAMA_CMAKE_ARGS: -DGGML_NATIVE=OFF'
assert_contains "${release_workflow}" 'scripts/package_release.sh'
assert_contains "${release_workflow}" 'the built Atelier frontend'
assert_not_contains "${release_workflow}" 'default 8B Qwen'

echo "ok"
