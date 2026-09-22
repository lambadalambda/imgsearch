# Make search modes and scope explicit

## Priority

P2 — Design recommendation.

## Summary

The tag pills look like refinements, but clicking one replaces the search. Counts and bare percentages add further ambiguity.

## Evidence

Twice in production: search “a quiet forest at dawn” → 48 results → click Tag · outdoor → the query clears, heading becomes outdoor, and URL becomes ?tag=outdoor. Media/sort controls disappear in search. Results show bare 42%/39% badges. The library header's image/video totals and visible-result count have different, unexplained scopes.

Reproduction / inspection steps:

1. Search for a quiet forest at dawn.
2. Click Tag · outdoor under the search box.
3. Observe that this replaces the query rather than refining it.

## Risks

A control visually attached to a search box is likely to be read as a filter on that search. “42%” can be mistaken for confidence or probability. Global totals beside the query can look like result totals.

## Requirements

- Choose a clear model: additive tag filters with removable chips, or explicitly labelled Browse tag shortcuts separate from the search controls.
- Preserve the query and filter scope visibly; make media filtering consistent between browsing and searching where supported.
- Label similarity as a relative match score, or replace the raw percentage with rank/relevance presentation. Do not present it as confidence.
- Distinguish Library totals from Visible results; use media/image/card consistently instead of pins and Quick collections for tag shortcuts.
- Provide a visible Search by image button that opens a picker, especially for mobile and keyboard users; paste/drop can remain shortcuts.

## Acceptance Criteria

- [x] Users can predict whether a Browse tag shortcut refines or replaces a search; counts and card match indicators identify their meaning and scope.
- [x] Add red→green coverage for behavior changes; verify desktop and mobile UI.

## Notes

- Implemented in the main checkout; awaiting parent review. No commit, index/archive update, or deployment in this task.
- Chosen model: a separate **Browse tags — Replaces current query** area. A shortcut switches to the existing `?tag=…` mode and clears text search; browser Back restores the original query. Card tag buttons also name the replacement action for assistive technology and in their tooltip. No additive/backend search semantics were introduced.
- Header scope distinguishes text search, tag browsing, image search, and similar media. The query remains the heading/input. Desktop Library totals sit outside search. Library/tag/collection results show loaded **Visible results** and genuine scoped totals (including collection saved/hidden counts). Ranked text/similar/image APIs only count their capped returned batch, so those views instead say **Top results returned (not exhaustive)**.
- The current APIs support NSFW, but do not support type/sort parameters for text, tag, image, or similar searches. Those views explicitly show All media (and Relevance for ranked modes), retain NSFW, and explain that type/sort are Library-only. This avoids filtering just one loaded page or falsely implying persisted Library preferences apply. Collections show All saved media; Duplicates shows Images only.
- Card scores now read `Match 42/100`, with a visible results explanation, “Match scores are relative, not probabilities,” and a descriptive tooltip. The existing distance-to-score calculation is unchanged. Lightbox/Feed components are sibling-owned and untouched; Lightbox's existing percentage wording needs coordination in the parent review, rather than being silently changed here.
- **Search by image** is a visible button backed by an image-only file input. Enter (desktop) and Space (390px mobile) open a real browser file chooser; repeated selection works. Paste/drop remain supported and covered.
- TDD evidence: pure scope/score tests initially failed before the helper existed; a later duplicate-scope test failed on All media versus Images only, and a browser regression failed until the visible relative-score explanation existed. Final tests cover query→tag replacement and Back restoration, scoped counts, supported controls, non-percentage card badges, keyboard picker, paste/drop, and mobile/desktop layout.
- Review correction: the existing request key is also an eager render-scope guard. Library/text/tag/similar/image replacement requests clear previous cards and close/gate stale Lightbox content; current-scope append requests keep cards/Lightbox. Pending/failed replacements and ignored-abort late responses are covered for all five modes, plus same-NSFW query/order/image changes and failed/successful append retention. Collection race protections remain intact.
- Review red→green: old Library cards remained during a held safe replacement (`1 !== 0`); mobile tag navigation left focus hidden; the ranked-count helper test failed before implementation. All now pass. Focused browser commands: `ATELIER_SCOPE_ONLY=1 npm test` and `ATELIER_NAV_ONLY=1 npm test` after building.
- Validation: `mise run test:full` (Go, 53 frontend unit tests, script suite, built-asset validation, all smoke tests including UX07 races), `mise run check:frontend` (zero warnings/errors), formatting and `git diff --check`. Local fixture screenshots are under `/tmp/ux03-04/`; no production browser used.
- [ ] Parent follow-up, explicitly outside this change: inherited `annotationWatch` live-update NSFW filtering can differ from replacement-request filtering. Track/review that separately; this fix does not redesign annotation live updates.
- [HTML report, UX-04](../../docs/reviews/2026-09-22-atelier-ux-audit.html#UX-04) contains embedded screenshot evidence.

- Main nav/search implementation independently approved and full tests passed. Lightbox score consistency will be completed with UX06 before this issue is archived.
