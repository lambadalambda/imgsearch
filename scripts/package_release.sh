#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <target-label> [dist-dir]" >&2
  exit 1
fi

target_label="$1"
dist_dir="${2:-dist}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pkg_name="imgsearch-${target_label}"
pkg_root="${repo_root}/${dist_dir}/${pkg_name}"
llama_lib_dir="${repo_root}/deps/llama.cpp/build/bin"
sqlite_vector_dir="${repo_root}/tools/sqlite-vector"

has_macos_rpath() {
  local file="$1"
  local rpath="$2"
  otool -l "$file" | grep -A2 LC_RPATH | grep -F "path ${rpath} " >/dev/null 2>&1
}

delete_macos_rpath_if_present() {
  local file="$1"
  local rpath="$2"
  if has_macos_rpath "$file" "$rpath"; then
    install_name_tool -delete_rpath "$rpath" "$file"
  fi
}

rm -rf "${pkg_root}"
mkdir -p "${pkg_root}/lib" "${pkg_root}/models" "${pkg_root}/tools/sqlite-vector"

(
  cd "${repo_root}"
  go build -trimpath -ldflags='-s -w' -o "${pkg_root}/imgsearch" ./cmd/imgsearch
)

case "$(uname -s)" in
  Darwin)
    cp -a "${llama_lib_dir}"/lib*.dylib "${pkg_root}/lib/"
    ;;
  Linux)
    cp -a "${llama_lib_dir}"/lib*.so* "${pkg_root}/lib/"
    ;;
  *)
    echo "unsupported packaging host: $(uname -s)" >&2
    exit 1
    ;;
esac

cp -a "${sqlite_vector_dir}"/* "${pkg_root}/tools/sqlite-vector/"

cat > "${pkg_root}/README.txt" <<'EOF'
imgsearch rolling release

Contents:
- imgsearch
- lib/            bundled llama.cpp shared libraries
- tools/sqlite-vector/  bundled sqlite-vector extension
- models/         default model download location (auto-populated on first run)

First run:
1. Ensure libvips is installed on the system.
2. Run ./imgsearch
3. The default 8B Qwen GGUF files are downloaded automatically if missing.

Notes:
- The app auto-discovers sqlite-vector from ./tools/sqlite-vector/vector.
- Data is stored in ./data by default.
EOF

case "$(uname -s)" in
  Darwin)
    source_rpaths=(
      "${repo_root}/internal/embedder/llamacppnative/../../../deps/llama.cpp/build/bin"
      "${llama_lib_dir}"
    )
    for source_rpath in "${source_rpaths[@]}"; do
      delete_macos_rpath_if_present "${pkg_root}/imgsearch" "${source_rpath}"
    done
    install_name_tool -add_rpath "@executable_path/lib" "${pkg_root}/imgsearch"

    while IFS= read -r dylib; do
      for source_rpath in "${source_rpaths[@]}"; do
        delete_macos_rpath_if_present "${dylib}" "${source_rpath}"
      done
      install_name_tool -add_rpath "@loader_path" "${dylib}"
    done < <(find "${pkg_root}/lib" -type f -name '*.dylib')
    ;;
  Linux)
    patchelf --set-rpath '$ORIGIN/lib' "${pkg_root}/imgsearch"
    while IFS= read -r sofile; do
      patchelf --set-rpath '$ORIGIN' "${sofile}"
    done < <(find "${pkg_root}/lib" -type f -name '*.so*')
    ;;
esac

mkdir -p "${repo_root}/${dist_dir}"
tar -C "${repo_root}/${dist_dir}" -czf "${repo_root}/${dist_dir}/${pkg_name}.tar.gz" "${pkg_name}"

if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "${repo_root}/${dist_dir}/${pkg_name}.tar.gz" > "${repo_root}/${dist_dir}/${pkg_name}.tar.gz.sha256"
else
  shasum -a 256 "${repo_root}/${dist_dir}/${pkg_name}.tar.gz" > "${repo_root}/${dist_dir}/${pkg_name}.tar.gz.sha256"
fi

echo "packaged ${repo_root}/${dist_dir}/${pkg_name}.tar.gz"
