# Push browser history entries for in-app navigation

## Summary

Every in-app mode change (search, tag, similar, stats, back to library) writes the URL with `history.replaceState`, so no history entries are ever created. Pressing the browser Back button after clicking a tag chip or "Similar" leaves the site entirely instead of returning to the previous view. Verified live with Playwright: after clicking a quick-row tag chip (`?tag=woman`), `page.goBack()` navigated off-site.

The `popstate` listener in `stores.ts` and the back/forward comment in `SearchBar.svelte` show back/forward support was intended — the write side just never pushes.

## Requirements

- Mode changes that represent navigation (library ↔ search ↔ tag ↔ similar ↔ stats) must create history entries via `history.pushState`.
- Refinements of the same view (e.g. re-submitting the same search, initial URL normalization on load) should keep using `replaceState` to avoid history spam.
- Back/forward must restore the previous mode via the existing `popstate` listener without triggering an extra `pushState` (no loops, no duplicate entries).

## Acceptance Criteria

- From library, click a tag chip, then press Back → app returns to library mode, URL has no `tag` param, and the page does not leave the site.
- From a search, click "Similar" on a pin, press Back → returns to the search view with the query intact.
- Forward after Back restores the later view.
- UI smoke test covers the back-button flow (regression test added to `scripts/ui_smoke_test_atelier.mjs`).

## Notes

- Write side: `frontend/src/lib/stores.ts` (`writeURL`, `mode.subscribe`).
- A `popstate`-driven `mode.set` will re-enter the subscriber; the subscriber needs to know whether the change came from history navigation (skip pushing) or from user action (push).
- Found during the 2026-06-10 UI/UX review.
