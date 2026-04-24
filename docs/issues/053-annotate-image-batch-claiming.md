# 053: Make annotate_image Jobs Claimable in Batch Workers

## Priority

P2

## Status

Open.

## Summary

The batch worker claim path should not strand `annotate_image` jobs when the worker runs with batch processing enabled.

## Context

- `annotate_image` jobs are created by `EnsureAnnotationJobsForModel` and handled by the worker execution switch.
- The batch claim kind filter prioritizes and claims some job kinds for batched processing.
- If `annotate_image` is omitted from the batch claimable kinds, batch-only worker configurations can leave image annotation work pending while other job kinds progress.

## Risks

- Image annotation jobs can silently stall in batch worker configurations.
- Queue stats can show pending annotation work without the worker making progress.
- Future worker refactors can preserve the asymmetry if it is not captured explicitly.

## Acceptance Criteria

- [ ] Verify the current batch claim kind filter behavior for `annotate_image` jobs with a failing regression test.
- [ ] Make `annotate_image` jobs claimable by batch workers when an annotator is available.
- [ ] Preserve embed-before-annotate ordering so annotation does not run before image embeddings are complete.
- [ ] Add coverage for mixed available job kinds and the `allClaimedJobsAreKind` batch-embed guard.
- [ ] Ensure worker-only batch mode drains image annotation jobs.

## Related Files

- `internal/worker/queue.go`
- `internal/worker/queue_test.go`
- `internal/db/index_jobs.go`
