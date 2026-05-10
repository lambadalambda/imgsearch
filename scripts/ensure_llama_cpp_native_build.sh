#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/deps/llama.cpp/build"
bin_dir="${build_dir}/bin"
common_lib="${build_dir}/common/libcommon.a"
cmake_args_file="${build_dir}/.imgsearch-cmake-args"

cmake_args=(-DCMAKE_BUILD_TYPE=Release)
if [[ -n "${IMGSEARCH_LLAMA_CMAKE_ARGS:-}" ]]; then
  # Split like a shell command line so release builds can pass CMake flags
  # without forcing local developer builds to be portable and slower.
  extra_cmake_args=(${IMGSEARCH_LLAMA_CMAKE_ARGS})
  cmake_args+=("${extra_cmake_args[@]}")
fi
cmake_args_key="$(printf '%q ' "${cmake_args[@]}")"

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
  if [[ -f "${cmake_args_file}" ]]; then
    if [[ "$(<"${cmake_args_file}")" == "${cmake_args_key}" ]]; then
      exit 0
    fi
    rm -rf "${build_dir}"
  elif [[ -z "${IMGSEARCH_LLAMA_CMAKE_ARGS:-}" ]]; then
    exit 0
  else
    rm -rf "${build_dir}"
  fi
fi

cmake -S "${repo_root}/deps/llama.cpp" -B "${build_dir}" "${cmake_args[@]}"
cmake --build "${build_dir}" --target llama-server -j
printf '%s\n' "${cmake_args_key}" > "${cmake_args_file}"
