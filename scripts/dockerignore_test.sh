#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dockerignore="$repo_root/.dockerignore"

if grep -qx 'models/' "$dockerignore"; then
  echo ".dockerignore must anchor models/ as /models/ so llama.cpp source model files stay in the build context" >&2
  exit 1
fi

if ! grep -qx '/models/' "$dockerignore"; then
  echo ".dockerignore must exclude the top-level /models/ runtime model directory" >&2
  exit 1
fi

echo "ok"
