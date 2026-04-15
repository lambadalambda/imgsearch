# 004 Runtime Modes And Process Shape

## Priority

P1

## Status

Completed for the current runtime/process-shape scope.

- The app supports `-enable-annotations=false`, which skips Gemma asset resolution/loading and runs the API in an embed-only mode.
- The app now supports `-mode=all|api|worker`.
- `-mode=api -enable-annotations=false` runs the HTTP server without loading Gemma.
- `-mode=worker` runs the background worker without starting the HTTP server.
- Package wrappers and docs now describe the split-process API/worker setup.
- Dedicated embedding-worker / annotation-worker sub-modes are intentionally deferred until the pipeline moves to separate `embed_image` and `annotate_image` job kinds.

## Summary

The app currently auto-resolves and loads both the embedder and annotator in a single process by default. That makes it hard to run an embed-only API process or an annotation-only worker, even though that is likely the right shape for 24 GB GPUs and for the planned pipeline split.

## Why This Matters

- Search needs the embedder loaded for query-time `EmbedText`.
- Annotation is secondary and should be separable from search.
- Large annotator models compete with the embedder for VRAM.
- Upcoming performance work likely wants dedicated worker modes.

## Current Behavior

- `cmd/imgsearch/main.go`
- Default embedder assets are always resolved.
- If explicit annotator paths are not set, default annotator assets are resolved automatically.
- A separate annotator is then instantiated whenever annotator paths are present, which is now the default path.

## Desired Outcome

- Clear runtime modes such as:
- API/search process with embedder only,
- embedding worker,
- annotation worker,
- or a simple explicit switch to disable annotation loading.

## Suggested Approach

- Add an explicit annotation toggle, for example:
- `-enable-annotations=false`,
- or `-llama-native-annotator-variant=none`.
- Longer-term, consider explicit process modes that line up with `embed_image` and `annotate_image` work.

## Acceptance Criteria

- It is possible to start the API without loading Gemma.
- It is possible to run annotation separately from query-time search.
- Package wrappers and docs make the modes clear.
- Default behavior remains simple for local single-process use.

All acceptance criteria are satisfied by the current implementation.

## Related Files

- `cmd/imgsearch/main.go`
- `scripts/package_release.sh`
- `mise.toml`
- `docs/indexing-annotation-pipeline-notes.md`
