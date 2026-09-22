# Automatically consolidate strict duplicate images

## Summary

User reports that strict near-duplicate groups appear to be actual duplicates and asks whether they can be automatically deduplicated.

## Requirements

- First verify the strict threshold, grouping, and existing duplicate-delete behavior.
- Distinguish identical file hashes from perceptual matches; do not assume that a strict visual match guarantees interchangeable files.
- Agree keeper/deletion behavior before applying destructive changes to the production library.
- Preserve collection memberships and consider existing media references when consolidating duplicates.

## Acceptance Criteria

- Explain what strict matches mean and identify a safe bounded deduplication workflow.
- Confirm the desired automation and keeper policy with the user before implementing deletion behavior or running a production cleanup.
- Any approved behavior change has red→green regression tests, independent review, and explicit opt-in documenting the lossy unattended behavior.
- Concurrent imports select one direct keeper atomically; failed publication/commit preserves existing library data.
- Upgrades preserve collection memberships and metadata while preventing stale indexing results from reaching the replacement.

## Notes

- Investigation only initially; no media deleted and no background cleanup enabled.
- Verified: Atelier's Strict setting is a 64-bit dHash Hamming distance of at most 2; Exact only is distance 0, not byte or pixel equality. dHash reduces an image to brightness comparisons on a 9×8 grayscale grid, losing color and fine detail.
- Duplicate groups are transitive connected components, so two members of a group may be farther apart than the selected threshold. Video frames are excluded.
- Existing Keep largest selects highest pixel count (ties: lowest image ID) and deletes the other images after confirmation. It does not transfer their collection memberships.
- Upload already deduplicates identical file contents with SHA256.
- User selected unattended deduplication during imports, rather than the proposed preview-and-confirm bulk cleanup. No retroactive production sweep or deployment is part of this change.
- Implementation policy: compare each incoming standalone image directly against a keeper at Strict distance ≤2 (never delete a transitive component). Prefer highest pixel count and existing copy on ties; preserve collection memberships and user metadata. Videos and video-frame references must not be consolidated perceptually.
- Perceptual equality is not proof of identical content; document the trade-off and provide an opt-out. Verify concurrency and failure safety before enabling the workflow.

## Completion

- Implemented opt-in `-dedup-strict-images` for all standalone image uploads/imports. One direct candidate is selected inside the insert-first transaction; larger incoming copies receive a fresh ID and inherit metadata and collection memberships/order. Superseded disk files are deliberately retained. Existing groups are never swept.
- New-ID fencing avoids changing worker or thumbnail persistence; old all-model embeddings/jobs are removed atomically. Batch responses follow replacement chains and report `replaced_image_id`.
- Red→green upload regressions verified; `mise run test:full` passes (Go, 44 frontend units, script tests, and both browser smoke suites). `go test -race ./internal/upload ./internal/app ./cmd/imgsearch` passes. Go formatting and diff whitespace checks pass.
- Independent correctness/architecture review found no blockers; extra multi-candidate and paused-worker/thumbnail coverage is tracked in [integration coverage](expand-strict-import-deduplication-integration-coverage.md).
- Documented policy, API behavior, disk retention, opt-in, and new-ID link limitations in `docs/import-deduplication.md` and ADR-011. No deployment, production cleanup, or GitHub push performed.
