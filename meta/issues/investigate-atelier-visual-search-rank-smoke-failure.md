# Investigate Atelier visual search-rank smoke failure

## Summary

During verification of the standalone Fediverse importer, `mise run test:full` and a separate `mise run test:ui` both failed at `scripts/ui_smoke_test_atelier.mjs:1226`:

```text
expected visual top row to preserve search rank ["95%","91%","90%","86%"], got ["90%","95%","91%","86%"]
```

No frontend or browser smoke-test files changed in the importer work. A final `mise run test:full` rerun passed without any UI changes, so the failure is intermittent. The cause has not been investigated; this could be a layout regression or test timing/geometry issue.

## Requirements

- Reproduce the visual rank mismatch and distinguish a real masonry ordering bug from an unstable assertion.
- Preserve search relevance order without weakening coverage to conceal a regression.

## Acceptance Criteria

- A focused regression captures the root cause, if a code change is needed.
- `mise run test:ui` passes repeatedly with the expected visual rank order.

## Notes

- Kept separate from the importer to avoid unrelated UI changes.
- See the earlier [result-rank issue](065-preserve-result-rank-across-masonry-rows.md).
