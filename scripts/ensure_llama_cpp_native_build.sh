#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/deps/llama.cpp/build"
bin_dir="${build_dir}/bin"
common_lib="${build_dir}/common/libcommon.a"

case "$(uname -s)" in
  Darwin)
    expected_lib="${bin_dir}/libllama.dylib"
    ;;
  Linux)
    expected_lib="${bin_dir}/libllama.so"
    ;;
  *)
    echo "unsupported host platform: $(uname -s)" >&2
    exit 1
    ;;
esac

remove_foreign_build_if_present() {
  if [[ ! -d "${bin_dir}" ]]; then
    return
  fi

  local foreign=()
  shopt -s nullglob
  case "$(uname -s)" in
    Darwin)
      foreign=("${bin_dir}"/*.so "${bin_dir}"/*.so.*)
      ;;
    Linux)
      foreign=("${bin_dir}"/*.dylib)
      ;;
  esac
  shopt -u nullglob

  if (( ${#foreign[@]} == 0 )); then
    return
  fi

  printf 'removing foreign llama.cpp artifacts from %s\n' "${build_dir}" >&2
  rm -rf "${build_dir}"
}

remove_foreign_build_if_present

if [[ -f "${expected_lib}" && -f "${common_lib}" ]]; then
  exit 0
fi

cmake -S "${repo_root}/deps/llama.cpp" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --target llama-server -j
