# Avoid duplicate shared-frame annotation across video jobs

## Summary

Independent concurrency review found a non-blocking shared-frame edge case: concurrent video annotation jobs have separate evidence caches. If two videos reference the same previously unannotated frame image, both jobs can infer and write its annotation; the last write wins and the video summaries can use different evidence.

## Context and Risks

- `internal/worker/queue.go` caches evidence per video job, not across jobs.
- Now that remote video annotations can overlap, previously sequential cache misses can overlap too.
- Bounded by the configured annotation limit; does not bypass native serialization or the remote cap.
- Repeated requests can cost extra and produce inconsistent frame evidence; not a blocker for independent image/video job concurrency.

## Requirements

- Determine whether shared unannotated frames occur in representative libraries before adding coordination.
- If warranted, serialize/deduplicate generation for the same frame across jobs, preserving cancellation, reannotation intent, metadata edits, and ownership rules.
- Do not introduce a global lock that serializes unrelated annotations.

## Acceptance Criteria

- A regression with two videos sharing an unannotated frame proves one inference/result is used, including cancellation/failure paths.
- Independent images/videos still overlap up to the configured limit, and deliberate reannotation remains possible.

## Notes

- Deferred from the settings-concurrency change to keep its scope focused; no performance or cache redesign implemented here.
