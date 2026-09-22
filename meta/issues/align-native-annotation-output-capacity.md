# Align native annotation output capacity with configurable token budgets

## Summary

Independent review of configurable generation limits found an existing fixed 32 KiB native output buffer. Settings now allow up to 32768 generated tokens, but a byte ceiling is not equivalent to a token budget: verbose output can fail with `generation output exceeded buffer` before using the configured tokens.

## Context and risks

- `internal/embedder/llamacppnative/gemma_runtime_native.go` allocates the fixed output buffer; `bridge.cc` rejects output exceeding it.
- Native generation errors bypass the compact JSON parse retry. A larger token setting alone therefore cannot guarantee recovery.
- The remote backend also does not currently expose `finish_reason=length`, obscuring truncation diagnosis. No production failure was attributed to either condition by this source review.
- Keep any capacity increase bounded; do not replace the current limit with unbounded allocation.

## Acceptance Criteria

- Choose and document a bounded output-capacity policy consistent with supported token budgets.
- Add regression coverage for large native output and clear limit diagnostics.
- Distinguish model token truncation, context exhaustion and byte-buffer exhaustion where the backend supplies enough information; decide appropriate retry behavior separately.

## Notes

- Follow-up only; not implemented as part of Settings configurability.
