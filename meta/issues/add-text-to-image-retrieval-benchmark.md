# Add text-to-image retrieval benchmark

## Summary

The fixture retrieval test has 13 images and four queries. Every usable model passes it, so it cannot rank two embedding models. Add a reproducible benchmark that computes text-to-image Recall@K on a public captioned image set through the native embedder.

Parent: [Evaluate WeMM-Embedding-2B](evaluate-wemm-embedding-2b.md).

## Requirements

- Read a manifest of image paths and captions; download or prepare a fixed public subset (for example 1,000 images from a captioned dataset) outside the repository.
- Embed images and captions with the configured native model and report Recall@1/5/10 and image/text throughput.
- Keep the metric code pure and unit-tested; gate the model-backed run behind an env var like the existing integration tests.

## Acceptance Criteria

- One documented command produces Recall@K numbers for a given model path/mmproj/dimensions.
- Unit tests cover the metric and manifest parsing.

## Notes

- Dataset licence must allow local evaluation; do not commit dataset images.
