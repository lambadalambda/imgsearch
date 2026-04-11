#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: scripts/import_images.sh <source-dir> [api-base-url]"
  echo ""
  echo "examples:"
  echo "  scripts/import_images.sh ./fixtures/images"
  echo "  scripts/import_images.sh ./photos http://127.0.0.1:8080"
  echo ""
  echo "optional env:"
  echo "  IMGSEARCH_IMPORT_CONVERT=auto|never|vips (default: auto)"
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 1
fi

source_dir="$1"
api_base_url="${2:-${IMGSEARCH_API_URL:-http://127.0.0.1:8080}}"
upload_url="${api_base_url%/}/api/upload"
convert_mode="${IMGSEARCH_IMPORT_CONVERT:-auto}"

if [[ ! -d "$source_dir" ]]; then
  echo "source dir does not exist: $source_dir" >&2
  exit 1
fi

case "$convert_mode" in
  auto|never|vips) ;;
  *)
    echo "invalid IMGSEARCH_IMPORT_CONVERT value: $convert_mode" >&2
    exit 1
    ;;
esac

has_vips=0
if command -v vips >/dev/null 2>&1; then
  has_vips=1
fi

if [[ "$convert_mode" == "vips" && "$has_vips" -ne 1 ]]; then
  echo "IMGSEARCH_IMPORT_CONVERT=vips requires the 'vips' CLI in PATH" >&2
  exit 1
fi

if ! curl -sS -f -o /dev/null "${api_base_url%/}/healthz"; then
  echo "imgsearch API is not reachable at ${api_base_url%/} (healthz failed)" >&2
  exit 1
fi

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

convert_with_vips() {
  local src="$1"
  local dst="$2"
  vips copy "$src" "$dst[Q=90]" >/dev/null 2>&1
}

upload_once() {
  local src="$1"
  local resp_file="$2"
  local escaped_src="${src//\\/\\\\}"
  escaped_src="${escaped_src//\"/\\\"}"
  : > "$resp_file"
  curl -sS -o "$resp_file" -w "%{http_code}" -F "file=@\"${escaped_src}\"" "$upload_url"
}

total=0
created=0
duplicates=0
failed=0
converted=0

while IFS= read -r -d '' path; do
  total=$((total + 1))
  base_name="$(basename "$path")"
  ext="${base_name##*.}"
  ext="$(printf '%s' "$ext" | tr '[:upper:]' '[:lower:]')"

  should_convert=0
  if [[ "$ext" == "webp" || "$ext" == "avif" ]]; then
    should_convert=1
  fi

  upload_path="$path"

  if [[ "$should_convert" -eq 1 && "$convert_mode" == "vips" ]]; then
    converted_path="$tmp_dir/${total}-${base_name%.*}.jpg"
    if ! convert_with_vips "$path" "$converted_path"; then
      failed=$((failed + 1))
      echo "FAIL $path (vips conversion failed)"
      continue
    fi
    upload_path="$converted_path"
    converted=$((converted + 1))
  fi

  response_file="$tmp_dir/response-${total}.json"
  curl_err_file="$tmp_dir/curl-${total}.err"
  curl_error=""
  if ! http_code="$(upload_once "$upload_path" "$response_file" 2>"$curl_err_file")"; then
    http_code=""
    curl_error="$(tr '\n' ' ' < "$curl_err_file")"
  fi

  if [[ "$http_code" != "200" && "$http_code" != "201" && "$should_convert" -eq 1 && "$convert_mode" == "auto" && "$has_vips" -eq 1 && "$upload_path" == "$path" ]]; then
    converted_path="$tmp_dir/${total}-${base_name%.*}.jpg"
    if convert_with_vips "$path" "$converted_path"; then
      upload_path="$converted_path"
      converted=$((converted + 1))
      curl_error=""
      if ! http_code="$(upload_once "$upload_path" "$response_file" 2>"$curl_err_file")"; then
        http_code=""
        curl_error="$(tr '\n' ' ' < "$curl_err_file")"
      fi
    fi
  fi

  if [[ "$http_code" == "201" ]]; then
    created=$((created + 1))
    echo "OK   $path"
    continue
  fi
  if [[ "$http_code" == "200" ]]; then
    duplicates=$((duplicates + 1))
    echo "DUP  $path"
    continue
  fi

  failed=$((failed + 1))
  body=""
  if [[ -f "$response_file" ]]; then
    body="$(tr '\n' ' ' < "$response_file")"
  fi
  if [[ -n "$curl_error" ]]; then
    echo "FAIL $path (status=${http_code:-curl-error} curl=$curl_error body=$body)"
  else
    echo "FAIL $path (status=${http_code:-curl-error} body=$body)"
  fi
done < <(find "$source_dir" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" -o -iname "*.avif" \) -print0)

echo ""
echo "Import summary: total=$total created=$created duplicates=$duplicates converted=$converted failed=$failed"

if [[ "$total" -eq 0 ]]; then
  echo "No supported images found in $source_dir"
fi

if [[ "$failed" -gt 0 ]]; then
  exit 1
fi
