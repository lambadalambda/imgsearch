# Complete and deploy the remaining recommended UX sweep

## Summary

User approved the remaining recommended UX work and a deployment after completion. Keep changes topical and reviewable; ship together with the already committed legacy retirement and UX-07 Collections improvements.

## Scope / subissues

- [ ] UX-01/02: [Safe duplicate consolidation](preserve-collections-during-manual-duplicate-cleanup.md) and [honest match/keeper controls](clarify-perceptual-duplicate-safety-and-keeper-choice.md).
- [ ] UX-03/04: [Navigation/mobile chrome](simplify-atelier-navigation-and-mobile-chrome.md) and [search/filter scope](clarify-search-filter-scope-and-match-labels.md).
- [ ] UX-05: [Protect Settings drafts](protect-unsaved-atelier-settings-edits.md).
- [ ] UX-06: [Lightbox viewing/editing actions](simplify-lightbox-actions-and-edit-mode.md).
- [ ] UX-08: [Actionable status and error recovery](clarify-library-health-and-error-recovery.md).
- [ ] UX-10: [Feed return and media states](make-feed-exit-destination-and-loading-state-explicit.md).
- [ ] Deploy tested combined release to NAS CT107 and verify it.

## Requirements

- TDD, independent review before topical commits, formatting and full checks before commits.
- Preserve production data, settings, auth and media semantics. All editing/destructive browser tests use local fixtures only.
- Deployment: verified CT107 target, matching built Atelier assets, compatible Vulkan libraries, consistent SQLite snapshot and binary/link rollback. No Git push unless requested.
- Upload polish and unrelated performance/runtime followups are not included.

## Acceptance Criteria

- Scoped subissues completed with regression coverage and desktop/mobile checks.
- Production serves the final reviewed build and matching assets; existing data/settings preserved and worker/service health verified.
- Record deployment/rollback details; archive completed issues without deleting their notes.
