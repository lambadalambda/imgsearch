#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest_dir="${repo_root}/tools/sqlite-vector"

case "$(uname -s)" in
  Darwin)
    expected_lib="${dest_dir}/vector.dylib"
    foreign_lib="${dest_dir}/vector.so"
    ;;
  Linux)
    expected_lib="${dest_dir}/vector.so"
    foreign_lib="${dest_dir}/vector.dylib"
    ;;
  *)
    echo "unsupported host platform: $(uname -s)" >&2
    exit 1
    ;;
esac

if [[ ! -e "${expected_lib}" || -e "${foreign_lib}" || ! -e "${dest_dir}/vector" ]]; then
  "${repo_root}/scripts/setup_sqlite_vector.sh"
fi
