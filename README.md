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

## Notes

- The app binds to `127.0.0.1:8080` by default.
- Supported formats: JPEG, PNG, WEBP, and AVIF.
- Embedding uses the in-process `llama-cpp-native` runtime with the Qwen3-VL-Embedding-8B GGUF pair.
- The default Qwen embedding files and the default Gemma annotator files are downloaded automatically on first run when missing.
- Add `-enable-annotations=false` if you want to run the API without loading the Gemma annotation model.
- Add `-mode=api` or `-mode=worker` if you want to split the HTTP server and background worker into separate processes.
- Data is stored in `./data` by default.
- The UI includes uploads, indexing status, gallery browsing, text search, and similar-image search.

## For Developers

Development-focused setup, tasks, integration checks, and lower-level runtime notes are in:

- `docs/development.md`
- `docs/architecture.md`
- `docs/mvp-plan.md`
- `docs/decisions.md`
