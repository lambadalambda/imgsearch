#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

source_dir="$tmp_dir/source"
run_dir="$tmp_dir/run"
data_dir="$tmp_dir/data"
models_dir="$tmp_dir/models"
bin_dir="$tmp_dir/bin"
mkdir -p "$source_dir" "$run_dir" "$data_dir" "$models_dir" "$bin_dir"

printf 'jpg' >"$source_dir/a.jpg"
printf 'png' >"$source_dir/b.png"
printf 'webp' >"$source_dir/c.webp"
printf 'mp4' >"$source_dir/d.mp4"
printf 'webm' >"$source_dir/e.webm"
printf 'txt' >"$source_dir/ignore.txt"

podman_log="$tmp_dir/podman.log"
cat >"$bin_dir/podman" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"${IMGSEARCH_TEST_PODMAN_LOG:?missing log}"
case "${1:-}" in
  run)
    printf 'mock-container-id\n'
    ;;
  inspect)
    if [[ "${IMGSEARCH_TEST_PODMAN_RUNNING:-true}" == "true" ]]; then
      printf 'true\n'
    else
      printf 'false\n'
    fi
    ;;
  logs)
    printf 'mock logs\n'
    ;;
  stop)
    ;;
  *)
    ;;
esac
EOF
chmod +x "$bin_dir/podman"

cat >"$bin_dir/curl" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${IMGSEARCH_TEST_CURL_FAIL:-0}" == "1" ]]; then
  exit 7
fi
exit 0
EOF
chmod +x "$bin_dir/curl"

import_log="$tmp_dir/import.log"
mock_import="$tmp_dir/import.sh"
cat >"$mock_import" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'source=%s url=%s key=%s\n' "$1" "$2" "${IMGSEARCH_IMPORT_API_KEY:-}" >"${IMGSEARCH_TEST_IMPORT_LOG:?missing import log}"
python3 - "${IMGSEARCH_BENCH_DATA_DIR:?missing data dir}/imgsearch.sqlite" <<'PY'
import sqlite3
import os
import sys

db = sys.argv[1]
con = sqlite3.connect(db)
failed_jobs = int(os.environ.get("IMGSEARCH_TEST_IMPORT_FAILED_JOBS", "0"))
con.executescript('''
CREATE TABLE IF NOT EXISTS index_jobs(kind TEXT, state TEXT);
CREATE TABLE IF NOT EXISTS images(id INTEGER PRIMARY KEY);
CREATE TABLE IF NOT EXISTS videos(id INTEGER PRIMARY KEY);
CREATE TABLE IF NOT EXISTS video_frames(id INTEGER PRIMARY KEY);
DELETE FROM index_jobs;
DELETE FROM images;
DELETE FROM videos;
DELETE FROM video_frames;
INSERT INTO images(id) VALUES (1), (2);
INSERT INTO videos(id) VALUES (1);
INSERT INTO video_frames(id) VALUES (1), (2), (3);
''')
if failed_jobs:
    con.execute("INSERT INTO index_jobs(kind, state) VALUES ('annotate_image', 'failed')")
else:
    con.executescript('''
    INSERT INTO index_jobs(kind, state) VALUES ('embed_image', 'done'), ('annotate_image', 'done'), ('annotate_video', 'done');
    ''')
con.commit()
con.close()
PY
printf 'Import summary: total=3 created=3 duplicates=0 converted=0 failed=0\n'
EOF
chmod +x "$mock_import"

IMGSEARCH_TEST_PODMAN_LOG="$podman_log" \
  IMGSEARCH_TEST_IMPORT_LOG="$import_log" \
  IMGSEARCH_BENCH_PODMAN="$bin_dir/podman" \
  IMGSEARCH_BENCH_CURL="$bin_dir/curl" \
  IMGSEARCH_BENCH_IMPORT_SCRIPT="$mock_import" \
  IMGSEARCH_BENCH_CONTAINER_IMAGE="imgsearch:test" \
  IMGSEARCH_BENCH_ACCELERATOR="cpu" \
  IMGSEARCH_BENCH_DATA_DIR="$data_dir" \
  IMGSEARCH_BENCH_MODELS_DIR="$models_dir" \
  IMGSEARCH_BENCH_IMAGE_LIMIT=2 \
  IMGSEARCH_BENCH_VIDEO_LIMIT=1 \
  IMGSEARCH_BENCH_VIDEO_FRAME_COUNT=5 \
  IMGSEARCH_BENCH_API_KEY="benchmark-test-key" \
  IMGSEARCH_BENCH_POLL_SECONDS=1 \
  IMGSEARCH_BENCH_TIMEOUT_SECONDS=10 \
  IMGSEARCH_BENCH_MOUNT_LABEL="" \
  "$repo_root/scripts/benchmark_ingestion_container.sh" "$source_dir" "$run_dir" >/dev/null

if [[ ! -f "$run_dir/summary.json" ]]; then
  echo "expected summary.json" >&2
  exit 1
fi

python3 - "$run_dir/summary.json" "$run_dir/dataset-manifest.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
manifest = json.loads(Path(sys.argv[2]).read_text())
if summary["status"] != "passed":
    raise SystemExit(f"unexpected status {summary['status']}")
if manifest["images_selected"] != 2 or manifest["videos_selected"] != 1:
    raise SystemExit(f"unexpected manifest counts {manifest}")
if summary["media_counts"].get("images") != 2 or summary["media_counts"].get("videos") != 1:
    raise SystemExit(f"unexpected media counts {summary['media_counts']}")
PY

if ! grep -q -- "--name" "$podman_log"; then
  echo "expected podman run args to include --name" >&2
  cat "$podman_log" >&2
  exit 1
fi
if ! grep -q -- "-llama-native-use-gpu=false" "$podman_log"; then
  echo "expected CPU accelerator to disable llama native GPU" >&2
  cat "$podman_log" >&2
  exit 1
fi
if ! grep -q -- "-video-frame-count 5" "$podman_log"; then
  echo "expected video frame count experiment arg" >&2
  cat "$podman_log" >&2
  exit 1
fi
if ! grep -q -- "IMGSEARCH_API_KEY=benchmark-test-key" "$podman_log"; then
  echo "expected podman run args to pass explicit benchmark API key" >&2
  cat "$podman_log" >&2
  exit 1
fi
if ! grep -q "source=$run_dir/dataset url=http://127.0.0.1:18080 key=benchmark-test-key" "$import_log"; then
  echo "expected importer to receive staged dataset and benchmark URL" >&2
  cat "$import_log" >&2
  exit 1
fi

failed_jobs_run_dir="$tmp_dir/failed-jobs-run"
failed_jobs_data_dir="$tmp_dir/failed-jobs-data"
mkdir -p "$failed_jobs_run_dir" "$failed_jobs_data_dir"
failed_jobs_err="$tmp_dir/failed-jobs.err"
set +e
IMGSEARCH_TEST_PODMAN_LOG="$podman_log" \
  IMGSEARCH_TEST_IMPORT_LOG="$import_log" \
  IMGSEARCH_TEST_IMPORT_FAILED_JOBS=1 \
  IMGSEARCH_BENCH_PODMAN="$bin_dir/podman" \
  IMGSEARCH_BENCH_CURL="$bin_dir/curl" \
  IMGSEARCH_BENCH_IMPORT_SCRIPT="$mock_import" \
  IMGSEARCH_BENCH_CONTAINER_IMAGE="imgsearch:test" \
  IMGSEARCH_BENCH_ACCELERATOR="cpu" \
  IMGSEARCH_BENCH_DATA_DIR="$failed_jobs_data_dir" \
  IMGSEARCH_BENCH_MODELS_DIR="$models_dir" \
  IMGSEARCH_BENCH_IMAGE_LIMIT=1 \
  IMGSEARCH_BENCH_VIDEO_LIMIT=0 \
  IMGSEARCH_BENCH_POLL_SECONDS=1 \
  IMGSEARCH_BENCH_TIMEOUT_SECONDS=10 \
  IMGSEARCH_BENCH_MOUNT_LABEL="" \
  "$repo_root/scripts/benchmark_ingestion_container.sh" "$source_dir" "$failed_jobs_run_dir" >/dev/null 2>"$failed_jobs_err"
failed_jobs_status=$?
set -e
if [[ "$failed_jobs_status" == "0" ]]; then
  echo "expected benchmark to fail when indexing jobs failed" >&2
  exit 1
fi
if ! grep -q "benchmark completed with failed indexing jobs: 1" "$failed_jobs_err"; then
  echo "expected failed indexing jobs message" >&2
  cat "$failed_jobs_err" >&2
  exit 1
fi
python3 - "$failed_jobs_run_dir/summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
if summary["status"] != "failed":
    raise SystemExit(f"expected failed summary status, got {summary['status']}")
states = {(row["kind"], row["state"]): row["count"] for row in summary["jobs_by_kind_state"]}
if states.get(("annotate_image", "failed")) != 1:
    raise SystemExit(f"expected failed annotate_image job in summary, got {summary['jobs_by_kind_state']}")
PY

fail_run_dir="$tmp_dir/fail-run"
fail_data_dir="$tmp_dir/fail-data"
mkdir -p "$fail_run_dir" "$fail_data_dir"
fail_err="$tmp_dir/fail.err"
set +e
IMGSEARCH_TEST_PODMAN_LOG="$podman_log" \
  IMGSEARCH_TEST_IMPORT_LOG="$import_log" \
  IMGSEARCH_TEST_PODMAN_RUNNING=false \
  IMGSEARCH_TEST_CURL_FAIL=1 \
  IMGSEARCH_BENCH_PODMAN="$bin_dir/podman" \
  IMGSEARCH_BENCH_CURL="$bin_dir/curl" \
  IMGSEARCH_BENCH_IMPORT_SCRIPT="$mock_import" \
  IMGSEARCH_BENCH_CONTAINER_IMAGE="imgsearch:test" \
  IMGSEARCH_BENCH_ACCELERATOR="cpu" \
  IMGSEARCH_BENCH_DATA_DIR="$fail_data_dir" \
  IMGSEARCH_BENCH_MODELS_DIR="$models_dir" \
  IMGSEARCH_BENCH_IMAGE_LIMIT=1 \
  IMGSEARCH_BENCH_VIDEO_LIMIT=0 \
  IMGSEARCH_BENCH_POLL_SECONDS=1 \
  IMGSEARCH_BENCH_TIMEOUT_SECONDS=10 \
  IMGSEARCH_BENCH_MOUNT_LABEL="" \
  "$repo_root/scripts/benchmark_ingestion_container.sh" "$source_dir" "$fail_run_dir" >/dev/null 2>"$fail_err"
fail_status=$?
set -e
if [[ "$fail_status" == "0" ]]; then
  echo "expected benchmark to fail when container exits before health" >&2
  exit 1
fi
if ! grep -q "container exited before health endpoint became ready" "$fail_err"; then
  echo "expected fail-fast container exit message" >&2
  cat "$fail_err" >&2
  exit 1
fi

echo "ok"
