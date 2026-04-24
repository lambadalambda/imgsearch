# 039: Make Retry and Stats Media-Aware for Video Jobs

## Priority

P1

## Status

Open.

## Summary

Retry and stats endpoints should treat image jobs, video annotation jobs, and video transcript jobs consistently.

## Context

- `internal/jobs/http.go:61` queues missing image index and annotation jobs but not missing `transcribe_video` jobs.
- `internal/db/index_jobs.go:67` already provides `EnsureVideoTranscriptJobsForModel`.
- `internal/stats/http.go:93` joins failed jobs to `images`, excluding jobs with `video_id` such as `annotate_video` and `transcribe_video`.
- `stats.Response` currently exposes `images_total` but no first-class video counts or media-aware failure fields.

## Risks

- `/api/jobs/retry-failed` cannot repair missing transcript work.
- The UI can hide failed video jobs from recent failures.
- Queue health can look cleaner than reality when video work fails.

## Acceptance Criteria

- [ ] Include `EnsureVideoTranscriptJobsForModel` in retry-failed handling when transcription is enabled or when missing transcript jobs should be tracked.
- [ ] Add `VideoID` and media type information to `FailureItem`.
- [ ] Update the recent-failures query to left join both `images` and `videos`, populating whichever foreign key is non-null.
- [ ] Include failed `annotate_video` and `transcribe_video` jobs in stats responses.
- [ ] Consider adding `videos_total`, standalone image count, and video-frame image count to stats for clarity.
- [ ] Add tests for retrying missing transcript jobs and listing failed `annotate_video` and `transcribe_video` jobs.

## Related Files

- `internal/jobs/http.go`
- `internal/jobs/http_test.go`
- `internal/stats/http.go`
- `internal/stats/http_test.go`
- `internal/db/index_jobs.go`
