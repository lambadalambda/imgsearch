# 049: Extract Repeated Native Task Commands From mise.toml

## Priority

P3

## Status

Open.

## Summary

Native serve, worker, test, and benchmark tasks repeat long model/runtime flag blocks in `mise.toml`. Move shared command construction to scripts to reduce drift.

## Context

- `mise.toml` repeats native runtime flags for `serve`, `serve:api`, `serve:worker`, and annotator variants.
- Native integration test and benchmark tasks repeat model path, mmproj path, dimensions, GPU, context, batch, thread, and image limit environment setup.
- Docs already show drift around image max side defaults.

## Risks

- Runtime defaults drift between serve modes, smoke checks, docs, and benchmarks.
- New flags must be updated in several long one-line task definitions.
- Task definitions are hard to review because command construction is embedded in TOML strings.

## Acceptance Criteria

- [ ] Introduce shared scripts for native serve mode and native test/bench presets.
- [ ] Keep `mise.toml` tasks short and declarative.
- [ ] Preserve existing environment override behavior.
- [ ] Align or explicitly document image max side defaults across tasks and docs.
- [ ] Add a smoke check that exercises the shared script path.

## Related Files

- `mise.toml`
- `scripts/`
- `docs/development.md`
- `scripts/smoke_serve.sh`
