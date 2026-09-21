#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
package_script="$(<"${repo_root}/scripts/package_release.sh")"
build_script="$(<"${repo_root}/scripts/ensure_llama_cpp_native_build.sh")"
ci_workflow="$(<"${repo_root}/.github/workflows/ci.yml")"
release_workflow="$(<"${repo_root}/.github/workflows/rolling-release.yml")"
native_action="$(<"${repo_root}/.github/actions/native-deps/action.yml")"

assert_contains() {
  local haystack="$1"
  local needle="$2"
  if [[ "${haystack}" != *"${needle}"* ]]; then
    echo "expected workflow/script config to contain: ${needle}" >&2
    exit 1
  fi
}

assert_not_contains() {
  local haystack="$1"
  local needle="$2"
  if [[ "${haystack}" == *"${needle}"* ]]; then
    echo "expected workflow/script config not to contain: ${needle}" >&2
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
assert_contains "${package_script}" '-X imgsearch/internal/buildinfo.Commit=${build_commit}'
assert_contains "${package_script}" '${archive_name}.tar.gz'
assert_contains "${release_workflow}" 'removeArtifacts: true'
assert_contains "${build_script}" 'IMGSEARCH_LLAMA_CMAKE_ARGS'
assert_contains "${build_script}" 'IMGSEARCH_LLAMA_BUILD_JOBS'
# Shared native dependency action: cached libvips and llama.cpp builds.
assert_contains "${native_action}" 'vips-8.18.0.tar.xz'
assert_contains "${native_action}" 'DESTDIR=/tmp/vips-stage meson install'
assert_contains "${native_action}" 'actions/cache@v4'
assert_contains "${native_action}" 'path: deps/llama.cpp/build'
assert_contains "${native_action}" 'git rev-parse HEAD:deps/llama.cpp'
assert_contains "${native_action}" 'IMGSEARCH_LLAMA_CMAKE_ARGS'
assert_contains "${native_action}" 'IMGSEARCH_LLAMA_BUILD_JOBS'
assert_not_contains "${native_action}" 'libvips-dev'
# CI: native deps plus formatting, vet, race, frontend, script, and smoke checks.
assert_contains "${ci_workflow}" 'uses: ./.github/actions/native-deps'
assert_contains "${ci_workflow}" 'extra-linux-packages: ffmpeg'
assert_contains "${ci_workflow}" 'gofmt -l'
assert_contains "${ci_workflow}" 'go vet ./...'
assert_contains "${ci_workflow}" 'go test -race ./...'
assert_contains "${ci_workflow}" 'npm run check'
assert_contains "${ci_workflow}" 'npm test'
assert_contains "${ci_workflow}" 'npm run build'
assert_contains "${ci_workflow}" 'scripts/*_test.sh'
assert_contains "${ci_workflow}" 'npx playwright install --with-deps chromium'
assert_not_contains "${ci_workflow}" 'libvips-dev'
# Release: same action with portable CPU flags on Linux.
assert_contains "${release_workflow}" 'actions/setup-node@v4'
assert_contains "${release_workflow}" 'uses: ./.github/actions/native-deps'
assert_contains "${release_workflow}" 'extra-linux-packages: patchelf'
assert_contains "${release_workflow}" '-DGGML_NATIVE=OFF'
assert_contains "${release_workflow}" 'scripts/package_release.sh'
assert_contains "${release_workflow}" 'the built Atelier frontend'
assert_not_contains "${release_workflow}" 'default 8B Qwen'

echo "ok"
