# Normalize near-duplicate annotation tags

## Summary

The annotator emits near-duplicate tag variants that surface side by side in the UI: the quick-row showed both "Tag · outdoor" and "Tag · outdoors" as adjacent chips. Splitting one concept across variants dilutes tag search and wastes quick-row slots.

## Requirements

- Normalize tags at annotation/ingestion time: lowercase (already done?), trim, and fold trivial singular/plural and whitespace/hyphen variants into one canonical form.
- Migration or backfill pass for existing tags so old and new media agree.
- Tag search continues to match media tagged with pre-normalization variants.

## Acceptance Criteria

- Ingesting media whose annotator output contains "outdoor" and "outdoors" stores a single canonical tag.
- Quick-row/tag-cloud no longer lists trivial variants separately after backfill.
- Go tests cover the normalization rules.

## Notes

- Backend issue (annotation pipeline + tag storage); UI is just where it shows.
- Keep rules conservative — only fold clearly-trivial variants; do not stem aggressively.
- Found during the 2026-06-10 UI/UX review.
