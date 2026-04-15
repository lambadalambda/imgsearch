# 010 Doc Drift After Native-Only Cutover

## Priority

P3

## Status

Completed.

- Architecture and decision docs now describe the in-process `llama-cpp-native` runtime rather than an endpoint-oriented embedding path.
- The architecture doc no longer references the nonexistent `vector_index_entries` table.
- The `Unreleased` changelog no longer carries stale sqlite-ai entries.
- Release-facing docs now mention both the default Qwen embedder download and the default Gemma annotator download behavior.

## Summary

Some design and release notes still describe older runtime shapes even though the repo is now native-only for embedding. This is not a runtime bug, but it is worth cleaning up so future performance work is guided by the current system rather than historical designs.

## Why This Matters

- Architecture docs still influence implementation decisions.
- Stale change notes make it harder to understand what the repo actually does now.
- Native-only runtime expectations should be clear in both developer docs and release-facing docs.

## Current Drift

- `docs/architecture.md`
- Still describes a generic pluggable embedding adapter and a local runtime endpoint as the MVP default.
- Still describes sqlite-vector mirroring in a way that should be reconciled with the current persistence story.
- `docs/decisions.md`
- Still reflects the older endpoint-oriented runtime framing.
- `CHANGELOG.md`
- Unreleased items include stale sqlite-ai references.

## Desired Outcome

- Docs accurately describe the current native-only system.
- The longer-term pipeline split is linked clearly from current architecture docs.
- Historical notes remain useful without misleading readers about the active runtime.

## Acceptance Criteria

- Architecture and decisions docs match the current embedding/runtime model.
- Changelog entries describe real current work.
- Release-facing docs mention both default embedder and default annotator behavior accurately.

All acceptance criteria are satisfied by the current implementation.

## Related Files

- `docs/architecture.md`
- `docs/decisions.md`
- `docs/development.md`
- `README.md`
- `CHANGELOG.md`
