#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: scripts/benchmark_ingestion_container.sh <source-media-dir> [run-dir]

Runs an imgsearch ingestion benchmark against a Podman container. The source
directory is staged into a fixed-size dataset, uploaded through the real HTTP
API, and the script waits for indexing jobs to drain in the mounted SQLite DB.

Common env:
  IMGSEARCH_BENCH_CONTAINER_IMAGE=imgsearch:cuda
  IMGSEARCH_BENCH_ACCELERATOR=cuda|cpu        default: cuda
  IMGSEARCH_BENCH_MODELS_DIR=./models         host model directory
  IMGSEARCH_BENCH_IMAGE_LIMIT=50              images copied into dataset
  IMGSEARCH_BENCH_VIDEO_LIMIT=5               videos copied into dataset
  IMGSEARCH_BENCH_PORT=18080                  host port for benchmark API
  IMGSEARCH_BENCH_API_KEY=...                 explicit API key for container and importer
  IMGSEARCH_BENCH_TIMEOUT_SECONDS=7200        wait timeout for jobs
  IMGSEARCH_BENCH_ANNOTATOR_IMAGE_MAX_SIDE=N  optional experiment knob
  IMGSEARCH_BENCH_VIDEO_FRAME_COUNT=N         optional sampled frames per video
  IMGSEARCH_BENCH_APP_ARGS='...'              extra imgsearch args
  IMGSEARCH_BENCH_KEEP_CONTAINER=1            leave container running
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$1"
if [[ ! -d "$source_dir" ]]; then
  echo "source media dir does not exist: $source_dir" >&2
  exit 1
fi
source_dir="$(cd "$source_dir" && pwd)"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${2:-${IMGSEARCH_BENCH_RUN_DIR:-$repo_root/bench-results/ingestion-$timestamp}}"
mkdir -p "$run_dir"
run_dir="$(cd "$run_dir" && pwd)"

podman_bin="${IMGSEARCH_BENCH_PODMAN:-podman}"
curl_bin="${IMGSEARCH_BENCH_CURL:-curl}"
import_script="${IMGSEARCH_BENCH_IMPORT_SCRIPT:-$repo_root/scripts/import_images.sh}"
container_image="${IMGSEARCH_BENCH_CONTAINER_IMAGE:-imgsearch:cuda}"
accelerator="${IMGSEARCH_BENCH_ACCELERATOR:-cuda}"
container_name="${IMGSEARCH_BENCH_CONTAINER_NAME:-imgsearch-bench-$timestamp}"
port="${IMGSEARCH_BENCH_PORT:-18080}"
api_key="${IMGSEARCH_BENCH_API_KEY:-${IMGSEARCH_API_KEY:-imgsearch-bench-default-api-key}}"
image_limit="${IMGSEARCH_BENCH_IMAGE_LIMIT:-50}"
video_limit="${IMGSEARCH_BENCH_VIDEO_LIMIT:-5}"
timeout_seconds="${IMGSEARCH_BENCH_TIMEOUT_SECONDS:-7200}"
poll_seconds="${IMGSEARCH_BENCH_POLL_SECONDS:-5}"
models_dir="${IMGSEARCH_BENCH_MODELS_DIR:-$repo_root/models}"
data_dir="${IMGSEARCH_BENCH_DATA_DIR:-$run_dir/data}"
dataset_dir="$run_dir/dataset"
manifest_path="$run_dir/dataset-manifest.json"
summary_path="$run_dir/summary.json"
container_log="$run_dir/container.log"
events_log="$run_dir/events.log"

case "$accelerator" in
  cuda|cpu) ;;
  *)
    echo "IMGSEARCH_BENCH_ACCELERATOR must be cuda or cpu, got: $accelerator" >&2
    exit 1
    ;;
esac

for numeric in "$image_limit" "$video_limit" "$timeout_seconds" "$poll_seconds" "$port"; do
  if ! [[ "$numeric" =~ ^[0-9]+$ ]]; then
    echo "benchmark numeric settings must be unsigned integers" >&2
    exit 1
  fi
done
if [[ -n "${IMGSEARCH_BENCH_VIDEO_FRAME_COUNT:-}" && ! "${IMGSEARCH_BENCH_VIDEO_FRAME_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "IMGSEARCH_BENCH_VIDEO_FRAME_COUNT must be a positive integer" >&2
  exit 1
fi

if [[ ! -d "$models_dir" ]]; then
  echo "model directory does not exist: $models_dir" >&2
  exit 1
fi
models_dir="$(cd "$models_dir" && pwd)"
mkdir -p "$data_dir" "$dataset_dir"

log_event() {
  local now
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s %s\n' "$now" "$*" | tee -a "$events_log"
}

stage_dataset() {
  IMGSEARCH_BENCH_SOURCE_DIR="$source_dir" \
    IMGSEARCH_BENCH_DATASET_DIR="$dataset_dir" \
    IMGSEARCH_BENCH_MANIFEST_PATH="$manifest_path" \
    IMGSEARCH_BENCH_IMAGE_LIMIT="$image_limit" \
    IMGSEARCH_BENCH_VIDEO_LIMIT="$video_limit" \
    python3 - <<'PY'
import json
import os
import shutil
from pathlib import Path

source = Path(os.environ["IMGSEARCH_BENCH_SOURCE_DIR"])
dataset = Path(os.environ["IMGSEARCH_BENCH_DATASET_DIR"])
manifest_path = Path(os.environ["IMGSEARCH_BENCH_MANIFEST_PATH"])
image_limit = int(os.environ["IMGSEARCH_BENCH_IMAGE_LIMIT"])
video_limit = int(os.environ["IMGSEARCH_BENCH_VIDEO_LIMIT"])

image_exts = {".jpg", ".jpeg", ".png", ".webp", ".avif"}
video_exts = {".gif", ".mp4", ".mov", ".webm", ".mkv"}

if dataset.exists():
    for child in dataset.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
dataset.mkdir(parents=True, exist_ok=True)

images = []
videos = []
for path in sorted(p for p in source.rglob("*") if p.is_file()):
    ext = path.suffix.lower()
    if ext in image_exts:
        images.append(path)
    elif ext in video_exts:
        videos.append(path)

selected = [("image", p) for p in images[:image_limit]] + [("video", p) for p in videos[:video_limit]]
manifest = {
    "source_dir": str(source),
    "image_limit": image_limit,
    "video_limit": video_limit,
    "images_available": len(images),
    "videos_available": len(videos),
    "images_selected": 0,
    "videos_selected": 0,
    "files": [],
}

for idx, (kind, path) in enumerate(selected):
    ext = path.suffix.lower()
    dst_name = f"{idx:04d}-{kind}{ext}"
    dst = dataset / dst_name
    shutil.copy2(path, dst)
    if kind == "image":
        manifest["images_selected"] += 1
    else:
        manifest["videos_selected"] += 1
    manifest["files"].append({"kind": kind, "source": str(path), "staged": dst_name, "bytes": dst.stat().st_size})

manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print(f"staged images={manifest['images_selected']} videos={manifest['videos_selected']} into {dataset}")
PY
}

mount_arg() {
  local host_path="$1"
  local container_path="$2"
  local mode="$3"
  local label="${IMGSEARCH_BENCH_MOUNT_LABEL:-Z}"
  if [[ -n "$label" ]]; then
    printf '%s:%s:%s,%s' "$host_path" "$container_path" "$mode" "$label"
  else
    printf '%s:%s:%s' "$host_path" "$container_path" "$mode"
  fi
}

run_container() {
  local -a run_args=(run -d --replace --name "$container_name")
  if [[ "$accelerator" == "cuda" ]]; then
    # shellcheck disable=SC2206
    local -a gpu_args=(${IMGSEARCH_BENCH_GPU_ARGS:---gpus=all})
    run_args+=("${gpu_args[@]}")
  fi

  local -a app_args=()
  if [[ "$accelerator" == "cpu" ]]; then
    app_args+=(
      -llama-native-use-gpu=false
      -llama-native-gpu-layers 0
      -llama-native-annotator-use-gpu=false
      -llama-native-annotator-gpu-layers 0
    )
  fi
  if [[ -n "${IMGSEARCH_BENCH_ANNOTATOR_IMAGE_MAX_SIDE:-}" ]]; then
    app_args+=(-llama-native-annotator-image-max-side "$IMGSEARCH_BENCH_ANNOTATOR_IMAGE_MAX_SIDE")
  fi
  if [[ -n "${IMGSEARCH_BENCH_VIDEO_FRAME_COUNT:-}" ]]; then
    app_args+=(-video-frame-count "$IMGSEARCH_BENCH_VIDEO_FRAME_COUNT")
  fi
  if [[ -n "${IMGSEARCH_BENCH_APP_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    app_args+=(${IMGSEARCH_BENCH_APP_ARGS})
  fi

  log_event "starting container name=$container_name image=$container_image accelerator=$accelerator"
  run_args+=(
    -p "127.0.0.1:${port}:8080"
    -e IMGSEARCH_ADDR=0.0.0.0:8080
    -e "IMGSEARCH_API_KEY=${api_key}"
    -v "$(mount_arg "$data_dir" /data rw)"
    -v "$(mount_arg "$models_dir" /models ro)"
    "$container_image"
  )
  if [[ ${#app_args[@]} -gt 0 ]]; then
    run_args+=("${app_args[@]}")
  fi
  "$podman_bin" "${run_args[@]}" >/dev/null
}

container_is_running() {
  [[ "$("$podman_bin" inspect -f '{{.State.Running}}' "$container_name" 2>/dev/null || true)" == "true" ]]
}

stop_container() {
  "$podman_bin" logs "$container_name" >"$container_log" 2>&1 || true
  if [[ "${IMGSEARCH_BENCH_KEEP_CONTAINER:-0}" == "1" ]]; then
    log_event "leaving container running name=$container_name"
    return
  fi
  "$podman_bin" stop "$container_name" >/dev/null 2>&1 || true
}

wait_health() {
  local deadline=$((SECONDS + timeout_seconds))
  local url="http://127.0.0.1:${port}/healthz"
  log_event "waiting for health url=$url"
  while (( SECONDS < deadline )); do
    if "$curl_bin" -fsS -o /dev/null "$url"; then
      log_event "health ready"
      return 0
    fi
    if ! container_is_running; then
      echo "container exited before health endpoint became ready: $container_name" >&2
      "$podman_bin" logs "$container_name" >&2 || true
      return 1
    fi
    sleep 1
  done
  echo "timed out waiting for health endpoint: $url" >&2
  return 1
}

active_job_count() {
  local db_path="$data_dir/imgsearch.sqlite"
  if [[ ! -f "$db_path" ]]; then
    printf '1'
    return 0
  fi
  python3 - "$db_path" <<'PY'
import sqlite3
import sys

db = sys.argv[1]
try:
    con = sqlite3.connect(db)
    cur = con.execute("SELECT COUNT(*) FROM index_jobs WHERE state IN ('pending', 'leased')")
    print(cur.fetchone()[0])
except Exception:
    print(1)
PY
}

failed_job_count() {
  local db_path="$data_dir/imgsearch.sqlite"
  if [[ ! -f "$db_path" ]]; then
    printf '0'
    return 0
  fi
  python3 - "$db_path" <<'PY'
import sqlite3
import sys

db = sys.argv[1]
try:
    con = sqlite3.connect(db)
    cur = con.execute("SELECT COUNT(*) FROM index_jobs WHERE state = 'failed'")
    print(cur.fetchone()[0])
except Exception:
    print(0)
PY
}

wait_jobs() {
  local deadline=$((SECONDS + timeout_seconds))
  log_event "waiting for indexing jobs to drain"
  while (( SECONDS < deadline )); do
    local active
    active="$(active_job_count)"
    log_event "active_jobs=$active"
    if [[ "$active" == "0" ]]; then
      local failed
      failed="$(failed_job_count)"
      if [[ "$failed" != "0" ]]; then
        echo "benchmark completed with failed indexing jobs: $failed" >&2
        return 1
      fi
      return 0
    fi
    sleep "$poll_seconds"
  done
  echo "timed out waiting for indexing jobs to drain" >&2
  return 1
}

write_summary() {
  local status="$1"
  IMGSEARCH_BENCH_SUMMARY_PATH="$summary_path" \
    IMGSEARCH_BENCH_MANIFEST_PATH="$manifest_path" \
    IMGSEARCH_BENCH_DB_PATH="$data_dir/imgsearch.sqlite" \
    IMGSEARCH_BENCH_STATUS="$status" \
    IMGSEARCH_BENCH_CONTAINER_IMAGE="$container_image" \
    IMGSEARCH_BENCH_ACCELERATOR="$accelerator" \
    IMGSEARCH_BENCH_CONTAINER_NAME="$container_name" \
    IMGSEARCH_BENCH_PORT="$port" \
    IMGSEARCH_BENCH_STARTED_AT="$started_at" \
    IMGSEARCH_BENCH_ENDED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    IMGSEARCH_BENCH_ELAPSED_SECONDS="$((SECONDS - start_seconds))" \
    python3 - <<'PY'
import json
import os
import sqlite3
from pathlib import Path

summary_path = Path(os.environ["IMGSEARCH_BENCH_SUMMARY_PATH"])
manifest_path = Path(os.environ["IMGSEARCH_BENCH_MANIFEST_PATH"])
db_path = Path(os.environ["IMGSEARCH_BENCH_DB_PATH"])

summary = {
    "status": os.environ["IMGSEARCH_BENCH_STATUS"],
    "started_at": os.environ["IMGSEARCH_BENCH_STARTED_AT"],
    "ended_at": os.environ["IMGSEARCH_BENCH_ENDED_AT"],
    "elapsed_seconds": int(os.environ["IMGSEARCH_BENCH_ELAPSED_SECONDS"]),
    "container_image": os.environ["IMGSEARCH_BENCH_CONTAINER_IMAGE"],
    "accelerator": os.environ["IMGSEARCH_BENCH_ACCELERATOR"],
    "container_name": os.environ["IMGSEARCH_BENCH_CONTAINER_NAME"],
    "port": int(os.environ["IMGSEARCH_BENCH_PORT"]),
    "dataset_manifest": str(manifest_path),
    "database": str(db_path),
    "jobs_by_kind_state": [],
    "media_counts": {},
}

if manifest_path.exists():
    summary["dataset"] = json.loads(manifest_path.read_text())

if db_path.exists():
    con = sqlite3.connect(db_path)
    try:
        for kind, state, count in con.execute("SELECT kind, state, COUNT(*) FROM index_jobs GROUP BY kind, state ORDER BY kind, state"):
            summary["jobs_by_kind_state"].append({"kind": kind, "state": state, "count": count})
        for table in ["images", "videos", "video_frames"]:
            try:
                count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                summary["media_counts"][table] = count
            except sqlite3.Error:
                pass
    finally:
        con.close()

summary_path.write_text(json.dumps(summary, indent=2) + "\n")
PY
}

started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
start_seconds="$SECONDS"
status="failed"
trap 'write_summary "$status"; stop_container' EXIT

stage_dataset | tee -a "$events_log"
run_container
wait_health

log_event "importing staged dataset"
IMGSEARCH_BENCH_DATA_DIR="$data_dir" \
  IMGSEARCH_IMPORT_API_KEY="$api_key" \
  "$import_script" "$dataset_dir" "http://127.0.0.1:${port}" | tee "$run_dir/import.log"

wait_jobs
status="passed"
write_summary "$status"
log_event "benchmark complete status=$status summary=$summary_path"
