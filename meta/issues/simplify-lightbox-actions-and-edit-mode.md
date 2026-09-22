# Separate viewing actions from editing actions

## Priority

P2 — Design recommendation.

## Summary

Save, Save to collection, Find similar, re-annotation and deletion compete in the same small action area—even during editing.

## Evidence

The desktop editor shows Save to collection beside Save, then Cancel / Find similar / Open original / Flag NSFW / Re-annotate / Delete. On mobile, a long hash filename and description push these actions below the visible sheet. Actual title/tag editing and saving succeeded in the sandbox.

Reproduction / inspection steps:

1. Open an image and choose Edit.
2. Compare the action area with the phone lightbox's initial viewport.

## Risks

Two different “save” concepts compete, and unrelated actions distract from finishing the edit. A filename that is mostly a hash has little reading value but consumes prime mobile space.

## Requirements

- During editing, emphasize Save changes and Cancel; move unrelated actions out of the edit flow.
- In viewing mode, prioritize Save to collection, Similar, and More; keep destructive/reprocessing actions in the latter.
- Make a compact action row sticky within the mobile sheet and collapse filename/technical fields under File details.
- Retain the current title/summary/description hierarchy and previous/next navigation—they are useful.

## Acceptance Criteria

- [ ] Save changes is unambiguous and reachable; core viewing actions remain easy to find without reading the entire description.
- [ ] Add red→green coverage for behavior changes; verify desktop and mobile UI.

## Notes

- Audit only; no fix implemented.
- [HTML report, UX-06](../../docs/reviews/2026-09-22-atelier-ux-audit.html#UX-06) contains embedded screenshot evidence.
