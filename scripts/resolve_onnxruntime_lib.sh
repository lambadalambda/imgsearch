#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${ONNXRUNTIME_GO_MODULE_DIR:-}" ]]; then
  module_dir="${ONNXRUNTIME_GO_MODULE_DIR}"
else
  go mod download github.com/yalue/onnxruntime_go >/dev/null
  module_dir="$(go list -m -f '{{.Dir}}' github.com/yalue/onnxruntime_go)"
fi

host_os="${IMGSEARCH_TEST_UNAME_S:-$(uname -s)}"
host_arch="${IMGSEARCH_TEST_UNAME_M:-$(uname -m)}"
ort_lib=""

case "$host_os" in
  Darwin)
    case "$host_arch" in
      arm64|aarch64)
        ort_lib="${module_dir}/test_data/onnxruntime_arm64.dylib"
        ;;
      *)
        echo "unsupported Darwin architecture for bundled onnxruntime_go test library: ${host_arch}" >&2
        exit 1
        ;;
    esac
    ;;
  Linux)
    case "$host_arch" in
      arm64|aarch64)
        ort_lib="${module_dir}/test_data/onnxruntime_arm64.so"
        ;;
      *)
        echo "unsupported Linux architecture for bundled onnxruntime_go test library: ${host_arch}" >&2
        exit 1
        ;;
    esac
    ;;
  *)
    echo "unsupported host platform: ${host_os}" >&2
    exit 1
    ;;
esac

if [[ -z "${ort_lib}" || ! -f "${ort_lib}" ]]; then
  echo "could not locate onnxruntime shared library in ${module_dir}/test_data" >&2
  exit 1
fi

printf '%s\n' "${ort_lib}"
