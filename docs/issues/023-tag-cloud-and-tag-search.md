# 023: Add Tag Cloud and Explicit Tag Search Flow

## Status: Completed

## Goal

Expose library tags as a first-class discovery surface and let users run direct searches by tag without relying on embedding-based text similarity.

## Problem

Tags were visible only per-card and there was no direct way to:

- discover high-frequency tags across the collection
- click a tag to search by it
- run exact multi-tag searches (e.g. "dog, outdoors")

## Desired Direction

- Show a compact tag cloud in the browsing workspace.
- Make cloud tags clickable and searchable.
- Support explicit tag search via a lightweight input for comma-separated tags.
- Keep semantic text search and similar-image search unchanged.

## Scope

- Add backend API for tag cloud aggregation.
- Add backend API for exact tag search (`any`/`all` mode).
- Integrate tag cloud + tag-search controls in web UI.
- Keep card/search rendering stable with mixed search sources.
- Add regression tests for API and static UI asset expectations.

## Scope Landed

- Added `GET /api/search/tag-cloud` to return ranked `{tag,count}` entries.
- Added `GET /api/search/tags` supporting repeated `tag` params (plus optional `mode=all|any`) and returning standard search results tagged with `search_source: "tag"`.
- Added a lightweight workspace-level `Tags` tab shell with lazy-loaded cloud rendering to keep tag discovery out of the masthead.
- Added clickable tag-cloud chips that run tag search and route output into the existing Results tab.
- Suppressed similarity match badges for `search_source: "tag"` results to avoid misleading score semantics.
- Added tag-search pagination (`limit` + `offset` + `total`) so users can page through all matching media.
- Expanded tag search to include both standalone images and videos (via tagged frame matches), with video results grouped per video.
- Added advanced-search tag restrictions with autocomplete suggestions and `all`/`any` matching mode for text search.
- Made card tag chips clickable so image/video tags can trigger tag search directly from grids.
- Added/updated backend tests in `internal/search/http_test.go`.
- Added/updated web UI static asset tests in `internal/webui/http_test.go`.

## Acceptance Criteria

- [x] Users can discover tags via a cloud surface
- [x] Users can click a tag to search by it
- [x] Users can run explicit multi-tag queries
- [x] Tag search results render in the existing Results tab
- [x] Backend and web UI tests cover the new functionality

## Notes

- Tag cloud ranking deduplicates per media unit (standalone image or grouped video) so repeated tagged frames from one video do not dominate counts.
