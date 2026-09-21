# Hot-swap the worker annotator from settings

## Summary

The worker holds a fixed `embedder.ImageAnnotator` set at composition time (`internal/app/compose.go:91-98`). Settings changes must take effect without a restart, in both single-process and split `-mode=api` / `-mode=worker` deployments.

## Requirements

- An `annotatorResolver` that owns the current backend: on each annotation job (or on a cheap version check every few seconds) it compares the persisted settings version with the one it built from, and rebuilds when they differ.
- Native backend: reuse `llamaModelSwitchboard` in `cmd/imgsearch`; variant changes replace the annotator loader closure, close the loaded annotator, and resolve the new variant's assets (downloading if missing, using `ensureDefaultLlamaNativeAnnotatorAssetsForVariant`). Explicit `-llama-native-annotator-*-path` flags still pin a custom native model and disable the variant selector in the UI.
- Remote backend: build an `openaicompat` annotator; the native annotator is unloaded to free memory.
- Swaps happen between jobs, never mid-job; an in-flight job finishes on the backend it started with.
- Expose the active backend and model in `/api/stats` (or `/api/settings`) so the UI can show "Annotating with: ...".
- Startup: persisted settings win over flag defaults after first seed; log the resolved backend.

## Acceptance Criteria

- Test: worker with a fake native annotator processes one job, settings switch to `openai` pointing at an `httptest` server, the next job hits the fake server.
- Test: switching variant triggers close of the old native annotator and construction of the new one exactly once.
- Split-process test or documented manual check: worker process observes a settings change made through the API process.

## Notes

- Subissue of [113](113-annotation-backend-settings-page.md). Depends on 115 and 116.
