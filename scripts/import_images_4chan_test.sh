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
mock_log="$tmp_dir/curl.log"
mock_state_dir="$tmp_dir/state"
mkdir -p "$mock_state_dir"
sleep_log="$tmp_dir/sleep.log"

cat >"$mock_bin/curl" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

log_file="${IMGSEARCH_TEST_CURL_LOG:-}"
if [[ -n "$log_file" ]]; then
  printf '%s\n' "$*" >>"$log_file"
fi

state_dir="${IMGSEARCH_TEST_STATE_DIR:-}"
if [[ -n "$state_dir" ]]; then
  mkdir -p "$state_dir"
fi

url=""
output_file=""
header_file=""
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
    -D)
      idx=$((idx + 1))
      header_file="${args[$idx]}"
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

emit_response() {
  local code="$1"
  local body="$2"
  local retry_after="${3:-}"

  if [[ -n "$header_file" ]]; then
    printf 'HTTP/1.1 %s\r\n' "$code" >"$header_file"
    if [[ -n "$retry_after" ]]; then
      printf 'Retry-After: %s\r\n' "$retry_after" >>"$header_file"
    fi
    printf '\r\n' >>"$header_file"
  fi

  if [[ -n "$output_file" ]]; then
    printf '%s' "$body" >"$output_file"
  else
    printf '%s' "$body"
  fi

  if [[ -n "$write_out" ]]; then
    printf '%s' "$code"
  fi
}

if [[ "$url" == "http://127.0.0.1:8080/healthz" ]]; then
  exit 0
fi

if [[ "$url" == "https://a.4cdn.org/v/thread/737156945.json" ]]; then
  first_try_flag="$state_dir/thread-json-first"
  if [[ -n "$state_dir" && ! -f "$first_try_flag" ]]; then
    touch "$first_try_flag"
    emit_response "429" "" "0"
    exit 0
  fi
  emit_response "200" '{"posts":[{"no":1,"tim":1111111111111,"ext":".jpg"},{"no":2,"tim":2222222222222,"ext":".webm"},{"no":3,"tim":3333333333333,"ext":".png"}]}'
  exit 0
fi

if [[ "$url" == "https://i.4cdn.org/v/1111111111111.jpg" || "$url" == "https://i.4cdn.org/v/2222222222222.webm" || "$url" == "https://i.4cdn.org/v/3333333333333.png" ]]; then
  if [[ "$url" == "https://i.4cdn.org/v/1111111111111.jpg" ]]; then
    first_try_flag="$state_dir/media-jpg-first"
    if [[ -n "$state_dir" && ! -f "$first_try_flag" ]]; then
      touch "$first_try_flag"
      emit_response "429" "" "0"
      exit 0
    fi
  fi
  emit_response "200" "img"
  exit 0
fi

if [[ "$url" == "http://127.0.0.1:8080/api/upload" ]]; then
  emit_response "201" '{"created":1}'
  exit 0
fi

echo "unexpected curl invocation: $*" >&2
exit 1
EOF
chmod +x "$mock_bin/curl"

cat >"$mock_bin/sleep" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

log_file="${IMGSEARCH_TEST_SLEEP_LOG:-}"
if [[ -n "$log_file" ]]; then
  printf '%s\n' "$*" >>"$log_file"
fi
EOF
chmod +x "$mock_bin/sleep"

set +e
output="$({
  IMGSEARCH_TEST_CURL_LOG="$mock_log" \
    IMGSEARCH_TEST_SLEEP_LOG="$sleep_log" \
    IMGSEARCH_TEST_STATE_DIR="$mock_state_dir" \
    IMGSEARCH_IMPORT_HTTP_RETRY_DELAY_SECONDS=0 \
    IMGSEARCH_IMPORT_4CHAN_MIN_DELAY_SECONDS=5 \
    IMGSEARCH_IMPORT_4CHAN_JITTER_SECONDS=2 \
    PATH="$mock_bin:$PATH" \
    "$repo_root/scripts/import_images.sh" "https://boards.4chan.org/v/thread/737156945" "http://127.0.0.1:8080"
} 2>&1)"
status=$?
set -e

if [[ "$status" -ne 0 ]]; then
  echo "expected thread import to succeed, got status $status" >&2
  printf '%s\n' "$output" >&2
  exit 1
fi

if [[ "$output" != *"Import summary: total=3 created=3 duplicates=0 converted=0 failed=0"* ]]; then
  echo "unexpected summary output" >&2
  printf '%s\n' "$output" >&2
  exit 1
fi

if ! grep -q "https://i.4cdn.org/v/1111111111111.jpg" "$mock_log"; then
  echo "expected full image download URL for jpg" >&2
  cat "$mock_log" >&2
  exit 1
fi

if ! grep -q "https://i.4cdn.org/v/3333333333333.png" "$mock_log"; then
  echo "expected full image download URL for png" >&2
  cat "$mock_log" >&2
  exit 1
fi

if ! grep -q "https://i.4cdn.org/v/2222222222222.webm" "$mock_log"; then
  echo "expected full media download URL for webm" >&2
  cat "$mock_log" >&2
  exit 1
fi

if [[ "$(grep -c "https://a.4cdn.org/v/thread/737156945.json" "$mock_log" || true)" -lt 2 ]]; then
  echo "expected thread json fetch to retry after 429" >&2
  cat "$mock_log" >&2
  exit 1
fi

if [[ "$(grep -c "https://i.4cdn.org/v/1111111111111.jpg" "$mock_log" || true)" -lt 2 ]]; then
  echo "expected media download to retry after 429" >&2
  cat "$mock_log" >&2
  exit 1
fi

if [[ ! -s "$sleep_log" ]]; then
  echo "expected pacing sleeps between media downloads" >&2
  cat "$mock_log" >&2
  exit 1
fi

sleep_count="$(wc -l < "$sleep_log" | tr -d ' ')"
if [[ "$sleep_count" -lt 2 ]]; then
  echo "expected at least two pacing sleep calls, got $sleep_count" >&2
  cat "$sleep_log" >&2
  exit 1
fi

while IFS= read -r delay; do
  if ! [[ "$delay" =~ ^[0-9]+$ ]] || [[ "$delay" -lt 5 ]] || [[ "$delay" -gt 7 ]]; then
    echo "expected pacing delay between 5 and 7 seconds, got '$delay'" >&2
    cat "$sleep_log" >&2
    exit 1
  fi
done < "$sleep_log"

if grep -q "s.jpg" "$mock_log"; then
  echo "did not expect thumbnail URL usage" >&2
  cat "$mock_log" >&2
  exit 1
fi

echo "ok"
