# 048: Extract Runtime Config From CLI Flag Parsing

## Priority

P2

## Status

Open.

## Summary

`cmd/imgsearch/main.go` defines many flags and immediately mixes parsing with runtime construction. Extract typed runtime configuration first so later app composition work can move in small steps.

## Context

- `cmd/imgsearch/main.go` handles flag declaration, parsing, defaulting, validation, and service construction in one function.
- Runtime mode, API key, native llama settings, annotator settings, Parakeet settings, and vector backend settings are all represented as local flag variables.
- A typed config is a low-risk prerequisite for later moving DB/vector/model construction into `internal/app`.

## Risks

- Startup validation is hard to test independently from service construction.
- Later composition refactors will be larger and riskier if configuration remains a pile of local variables.
- Defaults can drift between CLI, docs, and task scripts.

## Acceptance Criteria

- [ ] Extract parsed runtime configuration into a struct with explicit defaults.
- [ ] Keep config validation behavior equivalent to the current CLI path.
- [ ] Add tests for runtime mode, API key exposure, and native option validation through the config layer.
- [ ] Avoid moving DB/vector/model construction in this issue.
- [ ] Leave a clear handoff for `docs/issues/052-app-runtime-composition-layer.md`.

## Related Files

- `cmd/imgsearch/main.go`
- `cmd/imgsearch/*_test.go`
- `internal/app/bootstrap.go`
- `internal/app/bootstrap_test.go`
