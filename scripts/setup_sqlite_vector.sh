#!/usr/bin/env bash
set -euo pipefail

version="${1:-0.9.95}"
dest_dir="${2:-tools/sqlite-vector}"

os="$(uname -s | tr '[:upper:]' '[:lower:]')"
arch="$(uname -m)"

case "${os}/${arch}" in
  darwin/arm64)
    asset="vector-macos-arm64-${version}.tar.gz"
    ext=".dylib"
    ;;
  darwin/x86_64)
    asset="vector-macos-x86_64-${version}.tar.gz"
    ext=".dylib"
    ;;
  linux/arm64|linux/aarch64)
    asset="vector-linux-arm64-${version}.tar.gz"
    ext=".so"
    ;;
  linux/x86_64)
    asset="vector-linux-x86_64-${version}.tar.gz"
    ext=".so"
    ;;
  *)
    echo "unsupported platform: ${os}/${arch}" >&2
    exit 1
    ;;
esac

url="https://github.com/sqliteai/sqlite-vector/releases/download/${version}/${asset}"

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

mkdir -p "${dest_dir}"
rm -f "${dest_dir}/vector" "${dest_dir}/vector.dylib" "${dest_dir}/vector.so" "${dest_dir}/vector.dll"

archive="${tmp_dir}/vector.tar.gz"
curl -fL "${url}" -o "${archive}"
tar -xzf "${archive}" -C "${tmp_dir}"

source_file="${tmp_dir}/vector${ext}"
if [[ ! -f "${source_file}" ]]; then
  if [[ -f "${tmp_dir}/vector" ]]; then
    source_file="${tmp_dir}/vector"
  else
    echo "sqlite-vector archive did not contain expected library" >&2
    exit 1
  fi
fi

cp "${source_file}" "${dest_dir}/vector${ext}"

if [[ "${os}" == "darwin" ]]; then
  codesign --force --sign - "${dest_dir}/vector${ext}"
fi

ln -sf "vector${ext}" "${dest_dir}/vector"

echo "sqlite-vector installed at ${dest_dir}"
echo "set SQLITE_VECTOR_PATH=${dest_dir}/vector"
