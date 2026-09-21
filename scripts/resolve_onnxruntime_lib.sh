#!/usr/bin/env bash
# Print the path of an ONNX Runtime shared library for Parakeet video
# transcription, downloading the pinned official release into
# tools/onnxruntime/ when it is not cached yet.
#
# Environment:
#   IMGSEARCH_ONNXRUNTIME_LIB        explicit library path; printed as-is when it exists
#   IMGSEARCH_ONNXRUNTIME_VERSION    release to use (default: the version onnxruntime_go targets)
#   IMGSEARCH_ONNXRUNTIME_CACHE_DIR  cache root (default: <repo>/tools/onnxruntime)
#   IMGSEARCH_ONNXRUNTIME_DOWNLOAD   set to 0 to fail instead of downloading
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
version="${IMGSEARCH_ONNXRUNTIME_VERSION:-1.24.1}"
cache_dir="${IMGSEARCH_ONNXRUNTIME_CACHE_DIR:-${repo_root}/tools/onnxruntime}"
host_os="${IMGSEARCH_TEST_UNAME_S:-$(uname -s)}"
host_arch="${IMGSEARCH_TEST_UNAME_M:-$(uname -m)}"

if [[ -n "${IMGSEARCH_ONNXRUNTIME_LIB:-}" ]]; then
  if [[ ! -f "${IMGSEARCH_ONNXRUNTIME_LIB}" ]]; then
    echo "IMGSEARCH_ONNXRUNTIME_LIB does not exist: ${IMGSEARCH_ONNXRUNTIME_LIB}" >&2
    exit 1
  fi
  printf '%s\n' "${IMGSEARCH_ONNXRUNTIME_LIB}"
  exit 0
fi

# Map the host to the official release tarball and the library inside it.
case "${host_os}/${host_arch}" in
  Linux/x86_64)
    platform="linux-x64"
    lib_name="libonnxruntime.so.${version}"
    ;;
  Linux/arm64|Linux/aarch64)
    platform="linux-aarch64"
    lib_name="libonnxruntime.so.${version}"
    ;;
  Darwin/arm64|Darwin/aarch64)
    platform="osx-arm64"
    lib_name="libonnxruntime.${version}.dylib"
    ;;
  *)
    echo "no official ONNX Runtime ${version} build for ${host_os}/${host_arch}; build onnxruntime yourself and set IMGSEARCH_ONNXRUNTIME_LIB=/path/to/libonnxruntime, or run without video transcription" >&2
    exit 1
    ;;
esac

install_dir="${cache_dir}/${version}/${platform}"
lib_path="${install_dir}/lib/${lib_name}"

if [[ -f "${lib_path}" ]]; then
  printf '%s\n' "${lib_path}"
  exit 0
fi

if [[ "${IMGSEARCH_ONNXRUNTIME_DOWNLOAD:-1}" == "0" ]]; then
  echo "ONNX Runtime ${version} for ${platform} is not cached at ${lib_path} and downloads are disabled (IMGSEARCH_ONNXRUNTIME_DOWNLOAD=0)" >&2
  exit 1
fi

archive="onnxruntime-${platform}-${version}.tgz"
url="https://github.com/microsoft/onnxruntime/releases/download/v${version}/${archive}"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

echo "downloading ONNX Runtime ${version} (${platform}) from ${url}" >&2
curl -fsSL "${url}" -o "${tmp_dir}/${archive}"
tar -xzf "${tmp_dir}/${archive}" -C "${tmp_dir}"
mkdir -p "${install_dir}"
rm -rf "${install_dir}/lib"
cp -a "${tmp_dir}/onnxruntime-${platform}-${version}/lib" "${install_dir}/lib"
cp -a "${tmp_dir}/onnxruntime-${platform}-${version}/LICENSE" "${install_dir}/" 2>/dev/null || true

if [[ ! -f "${lib_path}" ]]; then
  echo "downloaded ONNX Runtime archive did not contain ${lib_name}" >&2
  exit 1
fi
printf '%s\n' "${lib_path}"
