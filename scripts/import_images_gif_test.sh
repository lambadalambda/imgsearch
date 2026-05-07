#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

mock_bin="$tmp_dir/bin"
mkdir -p "$mock_bin"
mock_log="$tmp_dir/calls.log"
source_dir="$tmp_dir/media"
mkdir -p "$source_dir"
printf 'gif89a' >"$source_dir/animated.gif"

cat >"$mock_bin/curl" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

log_file="${IMGSEARCH_TEST_CALL_LOG:-}"
if [[ -n "$log_file" ]]; then
  printf 'curl %s\n' "$*" >>"$log_file"
fi

url=""
output_file=""
write_out=""
args=("$@")
idx=0
while [[ "$idx" -lt "${#args[@]}" ]]; do
  arg="${args[$idx]}"
  case "$arg" in
    -o)
      idx=$((idx + 1))
      output_file="${args[$idx]}"
      ;;
    -w)
      idx=$((idx + 1))
      write_out="${args[$idx]}"
      ;;
  esac
  if [[ "$arg" == http://* || "$arg" == https://* ]]; then
    url="$arg"
  fi
  idx=$((idx + 1))
done

if [[ "$url" == "http://127.0.0.1:8080/healthz" ]]; then
  exit 0
fi

if [[ "$url" == "http://127.0.0.1:8080/api/upload" ]]; then
  if [[ -n "$output_file" ]]; then
    printf '{"created":1}' >"$output_file"
  fi
  if [[ -n "$write_out" ]]; then
    printf '201'
  fi
  exit 0
fi

echo "unexpected curl URL: $url" >&2
exit 1
EOF
chmod +x "$mock_bin/curl"

cat >"$mock_bin/ffmpeg" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

log_file="${IMGSEARCH_TEST_CALL_LOG:-}"
if [[ -n "$log_file" ]]; then
  printf 'ffmpeg %s\n' "$*" >>"$log_file"
fi

out="${@: -1}"
printf 'mp4' >"$out"
EOF
chmod +x "$mock_bin/ffmpeg"

set +e
output="$({
  unset IMGSEARCH_IMPORT_API_KEY IMGSEARCH_API_KEY
  IMGSEARCH_TEST_CALL_LOG="$mock_log" \
    IMGSEARCH_IMPORT_HTTP_RETRY_DELAY_SECONDS=0 \
    PATH="$mock_bin:$PATH" \
    "$repo_root/scripts/import_images.sh" "$source_dir" "http://127.0.0.1:8080"
} 2>&1)"
status=$?
set -e

if [[ "$status" -ne 0 ]]; then
  echo "expected gif import to succeed, got status $status" >&2
  printf '%s\n' "$output" >&2
  exit 1
fi

if [[ "$output" != *"Import summary: total=1 created=1 duplicates=0 converted=1 failed=0"* ]]; then
  echo "unexpected summary output" >&2
  printf '%s\n' "$output" >&2
  exit 1
fi

if ! grep -q "ffmpeg .*animated.gif" "$mock_log"; then
  echo "expected ffmpeg to convert gif" >&2
  cat "$mock_log" >&2
  exit 1
fi

if ! grep -q 'file=@".*/[0-9]-animated.mp4"' "$mock_log"; then
  echo "expected upload to use converted mp4" >&2
  cat "$mock_log" >&2
  exit 1
fi

echo "ok"

no_convert_dir="$tmp_dir/no-convert"
mkdir -p "$no_convert_dir"
printf 'gif89a' >"$no_convert_dir/skip.gif"

set +e
no_convert_output="$({
  unset IMGSEARCH_IMPORT_API_KEY IMGSEARCH_API_KEY
  IMGSEARCH_TEST_CALL_LOG="$mock_log" \
    IMGSEARCH_IMPORT_CONVERT=never \
    IMGSEARCH_IMPORT_HTTP_RETRY_DELAY_SECONDS=0 \
    PATH="$mock_bin:$PATH" \
    "$repo_root/scripts/import_images.sh" "$no_convert_dir" "http://127.0.0.1:8080"
} 2>&1)"
no_convert_status=$?
set -e

if [[ "$no_convert_status" -eq 0 ]]; then
  echo "expected gif import with conversion disabled to fail" >&2
  printf '%s\n' "$no_convert_output" >&2
  exit 1
fi
if [[ "$no_convert_output" != *"gif conversion disabled by IMGSEARCH_IMPORT_CONVERT=never"* ]]; then
  echo "expected conversion-disabled failure message" >&2
  printf '%s\n' "$no_convert_output" >&2
  exit 1
fi

missing_ffmpeg_bin="$tmp_dir/no-ffmpeg-bin"
mkdir -p "$missing_ffmpeg_bin"
cp "$mock_bin/curl" "$missing_ffmpeg_bin/curl"

set +e
missing_ffmpeg_output="$({
  unset IMGSEARCH_IMPORT_API_KEY IMGSEARCH_API_KEY
  IMGSEARCH_TEST_CALL_LOG="$mock_log" \
    IMGSEARCH_IMPORT_HTTP_RETRY_DELAY_SECONDS=0 \
    PATH="$missing_ffmpeg_bin:/usr/bin:/bin" \
    "$repo_root/scripts/import_images.sh" "$source_dir" "http://127.0.0.1:8080"
} 2>&1)"
missing_ffmpeg_status=$?
set -e

if [[ "$missing_ffmpeg_status" -eq 0 ]]; then
  echo "expected gif import without ffmpeg to fail" >&2
  printf '%s\n' "$missing_ffmpeg_output" >&2
  exit 1
fi
if [[ "$missing_ffmpeg_output" != *"gif conversion requires ffmpeg"* ]]; then
  echo "expected missing-ffmpeg failure message" >&2
  printf '%s\n' "$missing_ffmpeg_output" >&2
  exit 1
fi

echo "negative paths ok"
