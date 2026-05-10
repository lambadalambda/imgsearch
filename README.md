# imgsearch

`imgsearch` is a local-first image organization and similarity search app.

It runs as a small local web app that:
- indexes images into a SQLite-backed library,
- supports text-to-image and image-to-image search,
- keeps data on your machine,
- uses a built-in default multimodal embedding setup.

## Quick Start

### Use the Release

1. Download the latest archive from the GitHub `rolling` release for your system.
2. Extract it.
3. Install `libvips` on your machine.
4. Run `./imgsearch`.
5. Open `http://127.0.0.1:8080/`.

On first run, `imgsearch` downloads the default 8B embedding model into `./models/Qwen/` if it is missing, and also downloads the default Gemma annotator files when annotations are enabled.

### Import Images

With the app running, import a folder recursively:

```bash
./scripts/import_images.sh ~/Pictures/memes
```

Or any other folder:

```bash
./scripts/import_images.sh /path/to/your/images
```

The import script uploads supported images to the local app and indexing continues in the background.

You can also import full-size pictures and webms from a 4chan thread URL:

```bash
./scripts/import_images.sh https://boards.4chan.org/v/thread/737156945
```

For 4chan thread imports, the script pulls full files from `i.4cdn.org` (not thumbnails) and currently imports supported thread pictures plus `.webm` files.
If 4chan rate-limits requests (`HTTP 429`), the importer retries with `Retry-After` support.
If needed, tune retry behavior with `IMGSEARCH_IMPORT_HTTP_MAX_ATTEMPTS` and `IMGSEARCH_IMPORT_HTTP_RETRY_DELAY_SECONDS`.
By default, 4chan media downloads are paced (about every 5 seconds with jitter) to reduce rate-limit spikes.
`scripts/import_images.sh` / `mise run import-images` now sends an API key header by default.
Set `IMGSEARCH_IMPORT_API_KEY` (or `IMGSEARCH_API_KEY`) to override the built-in development key.

### Build From Source

If there is no release for your system:

1. Install Go, CMake, and `libvips`.
2. Initialize the llama.cpp submodule:
   - `git submodule update --init --recursive deps/llama.cpp`
3. Build llama.cpp runtime libraries:
   - `./scripts/ensure_llama_cpp_native_build.sh`
4. Install sqlite-vector:
   - `./scripts/setup_sqlite_vector.sh`
5. Run the app:
   - `go run ./cmd/imgsearch`

Cross-platform note:
- Keep host-native llama.cpp artifacts in `./deps/llama.cpp/build` only.
- If you build Linux artifacts from Docker on macOS, write them to an explicit separate directory such as `./build-artifacts/llama.cpp/linux-cuda13/` and pass `IMGSEARCH_LLAMA_LIB_DIR=/absolute/path/to/.../bin` when packaging.

### Podman + CUDA (Ubuntu Container)

If you want to run on a CUDA host through Podman while keeping an Ubuntu userspace, use `Containerfile.cuda`.

Quick path (host-accessible):

```bash
podman build -f Containerfile.cuda -t imgsearch:cuda .
podman run -d --name imgsearch --replace --gpus=all -p 8080:8080 \
  -e IMGSEARCH_ADDR=0.0.0.0:8080 \
  -e IMGSEARCH_API_KEY='replace-with-a-strong-token' \
  -v "$HOME/imgsearch-data:/data" \
  -v "$HOME/imgsearch-models:/models" \
  imgsearch:cuda
```

The container defaults to loopback-only bind (`127.0.0.1:8080`) for safer startup.
Set `IMGSEARCH_ADDR=0.0.0.0:8080` only when you intentionally want remote access, keep `IMGSEARCH_API_KEY` set, and place the service behind a trusted reverse proxy/TLS boundary.

Full instructions are in `docs/podman-cuda-ubuntu.md`.

## System Recommendations

The default profile targets a reasonably capable local machine: Qwen3-VL-Embedding-8B for search, GPU offload enabled when available, and the smaller Gemma annotator enabled. If that does not fit your machine, reduce memory in this order: disable annotations, reduce GPU layers, reduce batch size, reduce image size.

### Good GPU Or Unified Memory

Use this when you have a modern Apple Silicon system with enough unified memory, or a CUDA GPU with comfortable VRAM headroom.

```bash
./imgsearch
```

From a source checkout, the matching developer command is:

```bash
mise run serve
```

This gives the best out-of-box experience: image/video search, background indexing, and generated descriptions/tags.

### CPU-Only

Use this on machines without usable GPU acceleration, or when GPU drivers are unavailable. Indexing will be much slower, but the UI and already-indexed search remain usable.

```bash
./imgsearch \
  -enable-annotations=false \
  -llama-native-use-gpu=false \
  -llama-native-gpu-layers 0 \
  -llama-native-batch-size 128 \
  -llama-native-context-size 512 \
  -llama-native-image-max-side 320
```

If you really want CPU annotations too, remove `-enable-annotations=false` and add:

```bash
-llama-native-annotator-use-gpu=false -llama-native-annotator-gpu-layers 0
```

Expect annotations on CPU to be slow. For most CPU-only systems, search-only indexing is the practical profile.

### Low VRAM GPU

Use this when the default profile starts but crashes, gets killed, or reports GPU out-of-memory errors. The exact layer count is hardware dependent; start low and increase only after the queue drains reliably.

```bash
./imgsearch \
  -enable-annotations=false \
  -llama-native-gpu-layers 20 \
  -llama-native-batch-size 128 \
  -llama-native-context-size 512 \
  -llama-native-image-max-side 320
```

If this still fails, set `-llama-native-gpu-layers 0` or switch to the CPU-only command. If it is stable and you want more speed, try raising `-llama-native-gpu-layers` gradually.

For `mise run serve`, the equivalent embedder knobs are environment variables:

```bash
LLAMA_NATIVE_GPU_LAYERS=20 \
LLAMA_NATIVE_BATCH_SIZE=128 \
LLAMA_NATIVE_CONTEXT_SIZE=512 \
LLAMA_NATIVE_IMAGE_MAX_SIDE=320 \
mise run serve
```

Use direct `./imgsearch` or `go run ./cmd/imgsearch` when you also need flags such as `-enable-annotations=false`.

### Search-Only Server

Use this when you mainly care about similarity/text search and want to avoid loading the annotation model entirely.

```bash
./imgsearch -enable-annotations=false
```

This still embeds images and videos for search. It skips generated descriptions/tags, which is the largest memory and latency reduction.

### Large GPU And Better Annotations

The default annotator variant is the smaller `e4b` profile. On larger GPUs or high-memory unified-memory systems, you can try the 26B annotator for richer descriptions:

```bash
./imgsearch -llama-native-annotator-variant 26b
```

From a source checkout:

```bash
mise run "serve:8b:annotator-26b"
```

This is the heaviest local profile. If interactive search latency matters, run the UI/API without annotations and run a worker separately when you want to backfill annotations:

```bash
./imgsearch -mode=api -enable-annotations=false
./imgsearch -mode=worker -llama-native-annotator-variant 26b
```

Both processes must point at the same `-data-dir` if you split them.
On a single GPU, split mode can still increase total memory if API and worker run at the same time; if memory is tight, run the worker as a batch backfill job and stop it before latency-sensitive searches.

### Quick Tuning Reference

| Symptom | First change to try |
| --- | --- |
| GPU out of memory on startup | Add `-enable-annotations=false` |
| GPU out of memory while embedding | Lower `-llama-native-gpu-layers`, then lower `-llama-native-batch-size` |
| System memory pressure on CPU | Add `-enable-annotations=false` and lower `-llama-native-image-max-side` |
| Indexing is too slow but stable | Raise `-llama-native-gpu-layers` or `-llama-native-batch-size` one step at a time |
| Descriptions/tags are not needed | Keep `-enable-annotations=false` permanently |

## Notes

- The app binds to `127.0.0.1:8080` by default.
- Supported formats: JPEG, PNG, WEBP, and AVIF.
- Embedding uses the in-process `llama-cpp-native` runtime with the Qwen3-VL-Embedding-8B GGUF pair.
- The default Qwen embedding files and the default Gemma annotator files are downloaded automatically on first run when missing.
- Add `-enable-annotations=false` if you want to run the API without loading the Gemma annotation model.
- Add `-mode=api` or `-mode=worker` if you want to split the HTTP server and background worker into separate processes.
- `/api/*` routes are authenticated by default.
- Set `-api-key <token>` (or `IMGSEARCH_API_KEY`) to use your own key; when unset, the server falls back to a built-in development key and logs a startup warning.
- If you bind to a non-loopback address (for example `-addr 0.0.0.0:8080`), startup requires an explicit strong API key; the built-in development key is rejected.
- API clients can authenticate with `X-Imgsearch-API-Key: <token>` or `Authorization: Bearer <token>`.
- Multipart uploads to `/api/upload` keep partial-success semantics: each uploaded file returns either IDs/digest data or an `error`, mixed success/failure batches return `207 Multi-Status`, and oversized requests return `413 Payload Too Large`.
- Data is stored in `./data` by default.
- The UI includes uploads, indexing status, gallery browsing, text search, and similar-image search.

## For Developers

Development-focused setup, tasks, integration checks, and lower-level runtime notes are in:

- `docs/development.md`
- `docs/architecture.md`
- `docs/mvp-plan.md`
- `docs/decisions.md`
