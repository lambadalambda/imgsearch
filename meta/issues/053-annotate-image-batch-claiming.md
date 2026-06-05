# 053: Make annotate_image Jobs Claimable in Batch Workers

## Priority

P2

## Status

Done.

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

- [x] Verify the current batch claim kind filter behavior for `annotate_image` jobs with a failing regression test.
- [x] Make `annotate_image` jobs claimable by batch workers when an annotator is available.
- [x] Preserve embed-before-annotate ordering so annotation does not run before image embeddings are complete.
- [x] Add coverage for mixed available job kinds and the `allClaimedJobsAreKind` batch-embed guard.
- [x] Ensure worker-only batch mode drains image annotation jobs.

## Notes

- Issue 037 made `annotate_image` part of the batch claim kind filter when `Queue.Annotator` is available.
- Added regression coverage for the post-embed scenario where only `annotate_image` jobs remain, including an end-to-end `ProcessBatch` drain check.

## Related Files

- `internal/worker/queue.go`
- `internal/worker/queue_test.go`
- `internal/db/index_jobs.go`
