#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: scripts/import_images.sh <source-dir-or-4chan-thread-url> [api-base-url]"
  echo ""
  echo "examples:"
  echo "  scripts/import_images.sh ./fixtures/images"
  echo "  scripts/import_images.sh ./media http://127.0.0.1:8080"
  echo "  scripts/import_images.sh https://boards.4chan.org/v/thread/737156945"
  echo ""
  echo "optional env:"
  echo "  IMGSEARCH_IMPORT_CONVERT=auto|never|vips (default: auto)"
  echo "  IMGSEARCH_IMPORT_MAX_VIDEO_BYTES=<bytes> (default: 20971520, 20 MB)"
  echo "  IMGSEARCH_IMPORT_HTTP_MAX_ATTEMPTS=<n> (default: 6)"
  echo "  IMGSEARCH_IMPORT_HTTP_RETRY_DELAY_SECONDS=<seconds> (default: 2)"
  echo "  IMGSEARCH_IMPORT_4CHAN_USER_AGENT=<ua string> (default: desktop Chrome UA)"
  echo "  IMGSEARCH_IMPORT_4CHAN_MIN_DELAY_SECONDS=<seconds> (default: 5)"
  echo "  IMGSEARCH_IMPORT_4CHAN_JITTER_SECONDS=<seconds> (default: 2)"
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 1
fi

source_spec="$1"
api_base_url="${2:-${IMGSEARCH_API_URL:-http://127.0.0.1:8080}}"
upload_url="${api_base_url%/}/api/upload"
convert_mode="${IMGSEARCH_IMPORT_CONVERT:-auto}"
max_video_bytes="${IMGSEARCH_IMPORT_MAX_VIDEO_BYTES:-20971520}"
http_max_attempts="${IMGSEARCH_IMPORT_HTTP_MAX_ATTEMPTS:-6}"
http_retry_delay_seconds="${IMGSEARCH_IMPORT_HTTP_RETRY_DELAY_SECONDS:-2}"
fourchan_user_agent="${IMGSEARCH_IMPORT_4CHAN_USER_AGENT:-Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36}"
fourchan_min_delay_seconds="${IMGSEARCH_IMPORT_4CHAN_MIN_DELAY_SECONDS:-5}"
fourchan_jitter_seconds="${IMGSEARCH_IMPORT_4CHAN_JITTER_SECONDS:-2}"

source_mode=""
source_dir=""
thread_board=""
thread_id=""
if [[ "$source_spec" =~ ^https?://boards\.4chan\.org/([^/]+)/thread/([0-9]+)($|/.*$) ]]; then
  source_mode="4chan"
  thread_board="${BASH_REMATCH[1]}"
  thread_id="${BASH_REMATCH[2]}"
elif [[ -d "$source_spec" ]]; then
  source_mode="dir"
  source_dir="$source_spec"
else
  echo "source must be an existing directory or a 4chan thread URL: $source_spec" >&2
  exit 1
fi

case "$convert_mode" in
  auto|never|vips) ;;
  *)
    echo "invalid IMGSEARCH_IMPORT_CONVERT value: $convert_mode" >&2
    exit 1
    ;;
esac

if ! [[ "$max_video_bytes" =~ ^[0-9]+$ ]]; then
  echo "invalid IMGSEARCH_IMPORT_MAX_VIDEO_BYTES value: $max_video_bytes" >&2
  exit 1
fi

if ! [[ "$http_max_attempts" =~ ^[0-9]+$ ]] || [[ "$http_max_attempts" -lt 1 ]]; then
  echo "invalid IMGSEARCH_IMPORT_HTTP_MAX_ATTEMPTS value: $http_max_attempts" >&2
  exit 1
fi

if ! [[ "$http_retry_delay_seconds" =~ ^[0-9]+$ ]]; then
  echo "invalid IMGSEARCH_IMPORT_HTTP_RETRY_DELAY_SECONDS value: $http_retry_delay_seconds" >&2
  exit 1
fi

if ! [[ "$fourchan_min_delay_seconds" =~ ^[0-9]+$ ]]; then
  echo "invalid IMGSEARCH_IMPORT_4CHAN_MIN_DELAY_SECONDS value: $fourchan_min_delay_seconds" >&2
  exit 1
fi

if ! [[ "$fourchan_jitter_seconds" =~ ^[0-9]+$ ]]; then
  echo "invalid IMGSEARCH_IMPORT_4CHAN_JITTER_SECONDS value: $fourchan_jitter_seconds" >&2
  exit 1
fi

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

retry_delay_seconds() {
  local headers_file="$1"
  local attempt="$2"
  local retry_after=""

  if [[ -f "$headers_file" ]]; then
    retry_after="$(awk 'BEGIN{IGNORECASE=1} /^Retry-After:/ {gsub(/\r/, "", $2); print $2; exit}' "$headers_file")"
  fi

  if [[ "$retry_after" =~ ^[0-9]+$ ]]; then
    printf '%s' "$retry_after"
    return
  fi

  printf '%s' "$((http_retry_delay_seconds * attempt))"
}

next_fourchan_pace_delay_seconds() {
  local jitter=0
  if [[ "$fourchan_jitter_seconds" -gt 0 ]]; then
    jitter=$((RANDOM % (fourchan_jitter_seconds + 1)))
  fi
  printf '%s' "$((fourchan_min_delay_seconds + jitter))"
}

download_url_with_retry() {
  local url="$1"
  local output_path="$2"
  shift 2

  local attempt=0
  while (( attempt < http_max_attempts )); do
    attempt=$((attempt + 1))
    local header_file="$tmp_dir/http-headers-$$-${attempt}.txt"
    local curl_err_file="$tmp_dir/http-${attempt}.err"
    local http_code=""

    : > "$output_path"
    if ! http_code="$(curl -sS "$@" -D "$header_file" -o "$output_path" -w "%{http_code}" "$url" 2>"$curl_err_file")"; then
      if (( attempt >= http_max_attempts )); then
        return 1
      fi
      local sleep_seconds="$((http_retry_delay_seconds * attempt))"
      if [[ "$sleep_seconds" -gt 0 ]]; then
        sleep "$sleep_seconds"
      fi
      continue
    fi

    if [[ "$http_code" =~ ^2[0-9][0-9]$ ]]; then
      return 0
    fi

    if [[ "$http_code" == "429" && "$attempt" -lt "$http_max_attempts" ]]; then
      local sleep_seconds
      sleep_seconds="$(retry_delay_seconds "$header_file" "$attempt")"
      echo "RATE LIMIT $url (429), retrying in ${sleep_seconds}s (attempt ${attempt}/${http_max_attempts})" >&2
      if [[ "$sleep_seconds" -gt 0 ]]; then
        sleep "$sleep_seconds"
      fi
      continue
    fi

    return 1
  done

  return 1
}

fetch_4chan_thread_media_urls() {
  local board="$1"
  local thread="$2"
  local thread_json_url="https://a.4cdn.org/${board}/thread/${thread}.json"
  local thread_json

  if ! command -v python3 >/dev/null 2>&1; then
    echo "4chan thread import requires python3" >&2
    return 1
  fi

  local thread_json_file="$tmp_dir/4chan-thread-${board}-${thread}.json"
  if ! download_url_with_retry "$thread_json_url" "$thread_json_file"; then
    echo "failed to fetch 4chan thread JSON: $thread_json_url" >&2
    return 1
  fi

  thread_json="$(<"$thread_json_file")"

  if ! THREAD_JSON="$thread_json" python3 - "$board" <<'PY'
import json
import os
import sys

board = sys.argv[1]
payload = json.loads(os.environ["THREAD_JSON"])
allowed = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".webm"}
seen = set()
for post in payload.get("posts", []):
    tim = post.get("tim")
    ext = str(post.get("ext", "")).lower()
    if not tim or ext not in allowed:
        continue
    url = f"https://i.4cdn.org/{board}/{tim}{ext}"
    if url in seen:
        continue
    seen.add(url)
    print(url)
PY
  then
    echo "failed to parse 4chan thread JSON: $thread_json_url" >&2
    return 1
  fi
}

emit_source_paths() {
  if [[ "$source_mode" == "dir" ]]; then
    find "$source_dir" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" -o -iname "*.avif" -o -iname "*.mp4" -o -iname "*.mov" -o -iname "*.webm" -o -iname "*.mkv" \) -print0
    return
  fi

  local thread_download_dir="$tmp_dir/4chan-${thread_board}-${thread_id}"
  mkdir -p "$thread_download_dir"
  local thread_referer_url="https://boards.4chan.org/${thread_board}/thread/${thread_id}"

  local media_urls
  if ! media_urls="$(fetch_4chan_thread_media_urls "$thread_board" "$thread_id")"; then
    echo "failed to load media list from 4chan thread: ${source_spec}" >&2
    return 1
  fi

  if [[ -z "$media_urls" ]]; then
    echo "no supported full-size media found in thread: ${source_spec}" >&2
    return 0
  fi

  local media_index=0
  while IFS= read -r media_url; do
    [[ -z "$media_url" ]] && continue

    if [[ "$media_index" -gt 0 ]]; then
      local pace_delay_seconds
      pace_delay_seconds="$(next_fourchan_pace_delay_seconds)"
      if [[ "$pace_delay_seconds" -gt 0 ]]; then
        echo "PACE waiting ${pace_delay_seconds}s before next 4chan media download" >&2
        sleep "$pace_delay_seconds"
      fi
    fi

    local file_name="${media_url##*/}"
    local local_path="$thread_download_dir/$file_name"
    if ! download_url_with_retry "$media_url" "$local_path" -L -H "User-Agent: ${fourchan_user_agent}" -H "Referer: ${thread_referer_url}" -H "Accept: */*"; then
      echo "SKIP $media_url (download failed)" >&2
      media_index=$((media_index + 1))
      continue
    fi
    printf '%s\0' "$local_path"
    media_index=$((media_index + 1))
  done <<<"$media_urls"
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
  is_video=0
  if [[ "$ext" == "webp" || "$ext" == "avif" ]]; then
    should_convert=1
  fi
  if [[ "$ext" == "mp4" || "$ext" == "mov" || "$ext" == "webm" || "$ext" == "mkv" ]]; then
    is_video=1
  fi

  if [[ "$is_video" -eq 1 ]]; then
    file_size="$(stat -f %z "$path")"
    if [[ "$file_size" -gt "$max_video_bytes" ]]; then
      echo "SKIP $path (video exceeds max size ${max_video_bytes} bytes)"
      continue
    fi
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
done < <(emit_source_paths)

echo ""
echo "Import summary: total=$total created=$created duplicates=$duplicates converted=$converted failed=$failed"

if [[ "$total" -eq 0 ]]; then
  if [[ "$source_mode" == "dir" ]]; then
    echo "No supported media found in $source_dir"
  else
    echo "No supported media found in ${source_spec}"
  fi
fi

if [[ "$failed" -gt 0 ]]; then
  exit 1
fi
