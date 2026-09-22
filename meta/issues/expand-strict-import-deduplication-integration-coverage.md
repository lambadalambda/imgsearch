# Expand strict import deduplication integration coverage

## Summary

Independent review of unattended strict import deduplication found no correctness blockers, but identified two useful regression-test followups for its single-candidate and fresh-ID guarantees.

## Requirements

- Cover an incoming image larger than two differently sized direct matches. Only the largest existing candidate should be replaced; leave the other candidate's metadata and memberships untouched.
- Exercise replacement while an actual worker is paused in inference, then resume it; old results must not reach the replacement image.
- Exercise replacement while thumbnail generation is paused, then resume it; the old derivative must not be published for the new image ID.

## Acceptance Criteria

- Deterministic tests use synchronization barriers rather than sleeps or native inference.
- Tests pass under the race detector and assert surviving image IDs, jobs/embeddings, memberships, and derivative paths as appropriate.
- Keep production behavior unchanged unless a test demonstrates a real defect.

## Notes

- Existing coverage already tests direct-versus-transitive selection, tie-breaking, multiple service connections, metadata/collection preservation, rollback on publication/commit failure, and stale embedding rejection via enforced foreign keys.
- Static review verified the current worker and thumbnail paths are fenced by new image IDs. This is additional cross-package regression coverage, not a known corruption bug.
- See `internal/upload/strict_duplicates_test.go`, `internal/worker/queue.go`, and `internal/thumbnails/http.go`.
