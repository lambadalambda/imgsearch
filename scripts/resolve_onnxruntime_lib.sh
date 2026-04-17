#!/usr/bin/env bash
set -euo pipefail

go mod download github.com/yalue/onnxruntime_go >/dev/null
module_dir="$(go list -m -f '{{.Dir}}' github.com/yalue/onnxruntime_go)"

case "$(uname -s)" in
  Darwin)
    ort_lib="${module_dir}/test_data/onnxruntime_arm64.dylib"
    ;;
  Linux)
    ort_lib="$(ls "${module_dir}"/test_data/onnxruntime*.so 2>/dev/null | head -n 1 || true)"
    ;;
  *)
    echo "unsupported host platform: $(uname -s)" >&2
    exit 1
    ;;
esac

if [[ -z "${ort_lib}" || ! -f "${ort_lib}" ]]; then
  echo "could not locate onnxruntime shared library in ${module_dir}/test_data" >&2
  exit 1
fi

printf '%s\n' "${ort_lib}"
