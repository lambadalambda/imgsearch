# 050: Harden sqlite-vector Quantized Search Under Mixed Models

## Priority

P2

## Status

Open.

## Summary

Quantized sqlite-vector search scans global rows before filtering by model ID. The current app purges inactive model data, but the vector index contract should still behave correctly if mixed model rows exist.

## Context

- `internal/vectorindex/sqlitevector/index.go:96` caps quantized scan size from the active model count.
- `internal/vectorindex/sqlitevector/index.go:136` joins `vector_quantize_scan` results, then filters `WHERE ie.model_id = ?`.
- If other model rows are present, the global nearest rows can consume scan candidates and leave fewer active-model hits than requested.
- Existing startup behavior purges inactive model embeddings, which reduces urgency but does not make the index implementation robust in isolation.

## Risks

- Contract tests can pass under single-model data while mixed-model data under-returns results.
- Future multi-model support or retained model history can expose incorrect search results.
- Quantized and full-scan strategies can behave differently for the same query and limit.

## Acceptance Criteria

- [ ] Add a sqlite-vector test with rows from multiple model IDs that proves result count and ordering expectations.
- [ ] Include a concrete mixed-model case where a quantized search with `limit=10` against enough active-model rows returns 10 active-model results, not fewer.
- [ ] Oversample quantized candidates or retry with full scan when active-model hits are fewer than requested.
- [ ] Prefer a model-scoped vector scan strategy if sqlite-vector supports it cleanly.
- [ ] Keep search debug metadata accurate for retry/fallback behavior.
- [ ] Document any remaining single-active-model assumption in the vector backend contract.

## Related Files

- `internal/vectorindex/sqlitevector/index.go`
- `internal/vectorindex/sqlitevector/index_test.go`
- `internal/vectorindex/sqlitevector/index_integration_test.go`
- `internal/vectorindex/contract_test.go`
