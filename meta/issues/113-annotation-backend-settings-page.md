# Add a settings page for the annotation backend and model

## Summary

Descriptions, titles, summaries, and tags are produced by an in-process native Gemma annotator chosen at startup with flags (`-llama-native-annotator-variant`, model paths). Users should be able to open a Settings page in Atelier and choose where annotation ("descriptions and analysis") runs: the native `e4b` or `26b` model, or a remote OpenAI-compatible server (llama-server, Ollama, LM Studio, vLLM, OpenAI, OpenRouter, and similar) with a base URL, API key, and model name. Changes apply without restarting, existing annotations are kept, and a "Re-annotate all" action queues a full refresh on demand.

Search embeddings stay flag-driven and in-process; this issue covers annotation only.

## Requirements

- Persisted settings in SQLite so the API and worker processes share them, with flags seeding defaults on first run.
- Annotation backend selector: `native` (variant `e4b` or `26b`) or `openai` (base URL, API key, model, optional request timeout and concurrency).
- Prompts and response parsing shared between native and remote backends so both produce the same title/summary/description/tags/NSFW shape.
- Worker picks up backend changes at runtime; native variant switches unload and reload GGUF files through the existing switchboard.
- "Test connection" for the remote backend and a visible active-backend status.
- "Re-annotate all" button with confirmation.
- Atelier Settings page reachable from the Rail button that currently reads "Settings (soon)".

## Acceptance Criteria

- All subissues below are complete.
- Switching from native to a remote server in the UI causes the next annotation job to hit the remote server, verified by a handler test with a fake OpenAI-compatible server.
- Switching native variants reloads the model without restarting the process.
- The API key is never returned by `GET /api/settings`.

## Subissues

- [114](114-extract-shared-annotation-prompts-and-parsing.md) Extract shared annotation prompts and response parsing
- [115](115-add-persisted-settings-store-and-api.md) Add a persisted settings store and `/api/settings`
- [116](116-add-openai-compatible-remote-annotator.md) Add an OpenAI-compatible remote annotator
- [117](117-hot-swap-annotator-from-settings.md) Hot-swap the worker annotator from settings
- [118](118-add-reannotate-all-endpoint.md) Add a re-annotate-all endpoint
- [119](119-add-atelier-settings-page.md) Add the Atelier Settings page

## Notes

- Decisions (2026-09-21): OpenAI-compatible protocol only for the first version; keep existing annotations on switch; native and remote share one page.
- Seam: `internal/embedder` `ImageAnnotatorWithOptions`, `VideoFrameAnnotator`, `VideoAnnotator`; worker only depends on these.
