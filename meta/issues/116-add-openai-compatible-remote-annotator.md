# Add an OpenAI-compatible remote annotator

## Summary

Implement `embedder.ImageAnnotatorWithOptions`, `embedder.VideoFrameAnnotator`, and `embedder.VideoAnnotator` against the OpenAI chat-completions API with image input, so any compatible server (llama-server, Ollama, LM Studio, vLLM, OpenAI, OpenRouter) can produce annotations.

## Requirements

- New package `internal/embedder/openaicompat` with a client configured from `settings.AnnotationSettings.OpenAI`: base URL, API key (Bearer), model, timeout, concurrency limit.
- Image requests: resize through the existing libvips path to the configured annotator max side, encode as JPEG, send as a `data:image/jpeg;base64,...` `image_url` content part alongside the shared prompts from issue 114. Video annotation sends the text-only prompt built from frame evidence, plus the representative frame image.
- Request `response_format: {"type": "json_object"}` when the server accepts it; fall back silently if the server returns 400 for that field.
- Parse responses through the shared parsers; on parse failure, retry once with the shared retry prompts, mirroring the native flow.
- Bounded retries with backoff on 429 and 5xx, honouring `Retry-After`; context cancellation aborts in-flight requests.
- Never log the API key or full image payloads.

## Acceptance Criteria

- Tests with an `httptest` server assert the request shape (model, messages, image part, auth header), JSON parsing, retry-on-bad-JSON, 429 backoff, and cancellation.
- Manual check against a local llama-server or Ollama produces annotations that appear in the lightbox.

## Notes

- Subissue of [113](113-annotation-backend-settings-page.md). Depends on 114 and 115.
- Anthropic Messages API deliberately out of scope for the first version.
