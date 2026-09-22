# Keep NSFW-hidden views current during concurrent metadata updates

## Summary

Source reviews during the UX sweep found inherited visibility races outside the new request-scope guards. A live annotation update can mark a displayed card NSFW without removing it while NSFW is hidden. Duplicate discovery reads visible hashes and later hydrates cards without a common snapshot or a visibility recheck.

## Requirements

- Reconcile live card/lightbox visibility when annotation updates change NSFW classification.
- Ensure duplicate discovery does not hydrate newly hidden media from a stale visible-ID list.
- Preserve collection/detail filter guards and duplicate confirmation transaction revalidation; no broader search rewrite.

## Acceptance Criteria

- Deferred live-update and discovery-read tests prove newly hidden images cannot remain displayed/returned after classification changes.
- Cover current open lightbox and a non-preview video frame where relevant; do not merely hide content with CSS.

## Notes

- Review evidence: `frontend/src/lib/annotationWatch.ts` updates pins in place; `internal/images/duplicates.go` filters hash discovery before separate hydration.
- Manual consolidation preview/confirm revalidates visible group membership, so this is discovery/display safety rather than a transaction deleting hidden targets.
- No concurrent production repro was performed. Separate follow-up, not a scope expansion of UX03/04 or duplicate consolidation.
