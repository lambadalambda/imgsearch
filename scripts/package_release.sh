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
llama_lib_dir="${IMGSEARCH_LLAMA_LIB_DIR:-${repo_root}/deps/llama.cpp/build/bin}"
sqlite_vector_dir="${repo_root}/tools/sqlite-vector"

if [[ ! -d "${llama_lib_dir}" ]]; then
  echo "llama.cpp library directory not found: ${llama_lib_dir}" >&2
  echo "set IMGSEARCH_LLAMA_LIB_DIR when packaging from explicit cross-built artifacts" >&2
  exit 1
fi

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

is_linux_system_library() {
  local path="$1"
  case "$path" in
    ""|linux-vdso.so.*|linux-gate.so.*)
      return 0
      ;;
    /lib*/ld-linux*.so*|/lib*/libc.so.*|/lib*/libm.so.*|/lib*/libpthread.so.*|/lib*/libdl.so.*|/lib*/librt.so.*|/lib*/libresolv.so.*|/lib*/libutil.so.*)
      return 0
      ;;
  esac
  return 1
}

linux_list_shared_deps() {
  local file="$1"
  ldd "$file" | awk '
    /=>/ && $3 ~ /^\// { print $3 }
    /^[[:space:]]*\/[^[:space:]]+/ { print $1 }
  '
}

copy_linux_library_with_links() {
  local source_path="$1"
  local dest_dir="$2"
  local real_path
  local real_base
  local dest_real
  local source_base
  local soname

  real_path="$(readlink -f "$source_path")"
  real_base="$(basename "$real_path")"
  dest_real="${dest_dir}/${real_base}"
  source_base="$(basename "$source_path")"

  if [[ ! -e "$dest_real" ]]; then
    cp -a "$real_path" "$dest_real"
  fi

  soname="$(patchelf --print-soname "$real_path" 2>/dev/null || true)"
  if [[ -n "$soname" && "$soname" != "$real_base" && ! -e "${dest_dir}/${soname}" ]]; then
    ln -s "$real_base" "${dest_dir}/${soname}"
  fi

  if [[ "$source_base" != "$real_base" && ! -e "${dest_dir}/${source_base}" ]]; then
    ln -s "$real_base" "${dest_dir}/${source_base}"
  fi

  printf '%s\n' "$dest_real"
}

bundle_linux_runtime_deps() {
  local seen_file="$1"
  shift
  local queue=("$@")

  : > "$seen_file"

  while [[ ${#queue[@]} -gt 0 ]]; do
    local file="${queue[0]}"
    queue=("${queue[@]:1}")

    while IFS= read -r dep; do
      [[ -z "$dep" ]] && continue
      if is_linux_system_library "$dep"; then
        continue
      fi
      if [[ ! -e "$dep" ]]; then
        continue
      fi

      local real_dep
      real_dep="$(readlink -f "$dep")"
      if grep -Fqx "$real_dep" "$seen_file"; then
        continue
      fi
      printf '%s\n' "$real_dep" >> "$seen_file"

      local bundled_path
      bundled_path="$(copy_linux_library_with_links "$dep" "${pkg_root}/lib")"
      queue+=("$bundled_path")
    done < <(linux_list_shared_deps "$file")
  done
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
1. On Linux, run ./run.sh (or the preset wrappers) so bundled shared libraries are used.
2. On macOS, run ./imgsearch
3. The default 8B Qwen GGUF files and default Gemma annotator files are downloaded automatically if missing.
4. Add --enable-annotations=false if you want to skip loading the Gemma annotator.

Notes:
 - The app auto-discovers sqlite-vector from ./tools/sqlite-vector/vector.
 - Linux bundles include wrapper scripts that set LD_LIBRARY_PATH for the packaged shared libraries.
 - Data is stored in ./data by default.
 - Linux release archives also bundle libvips and its non-glibc runtime dependencies.
 - macOS release archives still expect libvips to be installed on the system.
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
    bundle_linux_runtime_deps "${pkg_root}/.linux-runtime-seen" "${pkg_root}/imgsearch" $(find "${pkg_root}/lib" -type f -name '*.so*' -print)
    rm -f "${pkg_root}/.linux-runtime-seen"
    patchelf --set-rpath '$ORIGIN/lib' "${pkg_root}/imgsearch"
    while IFS= read -r sofile; do
      patchelf --set-rpath '$ORIGIN' "${sofile}"
    done < <(find "${pkg_root}/lib" -type f -name '*.so*')

    for backend in libggml-cpu.so libggml-cuda.so; do
      if [[ -e "${pkg_root}/lib/${backend}" ]]; then
        ln -sf "lib/${backend}" "${pkg_root}/${backend}"
      fi
    done

    cat > "${pkg_root}/run.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="$script_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$script_dir/imgsearch" "$@"
EOF

    cat > "${pkg_root}/run-8b.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="$script_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SQLITE_VECTOR_PATH="$script_dir/tools/sqlite-vector/vector"
exec "$script_dir/imgsearch" -vector-backend sqlite-vector "$@"
EOF

    cat > "${pkg_root}/run-8b-annotator-26b.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="$script_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SQLITE_VECTOR_PATH="$script_dir/tools/sqlite-vector/vector"
exec "$script_dir/imgsearch" -vector-backend sqlite-vector -llama-native-annotator-variant 26b "$@"
EOF

    chmod +x "${pkg_root}/run.sh" "${pkg_root}/run-8b.sh" "${pkg_root}/run-8b-annotator-26b.sh"
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
