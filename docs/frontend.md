# Frontend architecture

**Atelier** at `/` is the only supported shell: Svelte 5 + Tailwind v4 +
Vite, with source in `frontend/` and build output embedded into the Go binary
from `internal/webui/atelier/dist/`. It uses the shared `/api/*` and `/media/*`
routes. Retired `/legacy` and `/legacy/*` URLs return **404**, not redirects.

## Atelier scope (MVP)

The first cut of the Atelier SPA covers:

- Library browse (paginated with `/api/images`)
- Text search via `/api/search/text` synced to `?q=` in the URL
- Find similar via `/api/search/similar` synced to `?similar=:image_id`
- Top-tags collection chips populated from `/api/search/tag-cloud`
- Live counts in the search-bar pills via `/api/stats`
- Pin lightbox for both images and videos (uses native `<video controls>`)
- NSFW filter toggled in the header (passes `include_nsfw=1`)
- Mobile layout with the rail collapsing to a top dock and the masonry
  collapsing to a single column
- Per-pin actions: Find similar, Flag NSFW (optimistic), Re-annotate, Delete
- Tag chip search via `/api/search/tags` synced to `?tag=`
- Pagination via the `Load more` button (offset bumps in 48-pin pages)
- Saved image/video collections (`?view=collections` / `?view=collection&collection_id=123`), with create/rename/delete, mixed-media browsing, and a shared save/remove picker from cards, lightbox, and the current Feed video. Collections reference originals rather than copying media; see [collections](collections.md).
- Upload modal (multi-select + drag-drop) hitting `POST /api/upload` with
  per-file row states (pending → uploading → created/duplicate/failed)
- Similar-video Feed overlay: seeded by any video pin's "Feed" corner
  action or by the Rail Feed button, which samples a random playable video
  from `/api/videos`; the overlay drives `/api/search/similar-videos` with
  session-local prefer/avoid tag preferences, supports keyboard
  (↑↓/Esc/Space) and vertical swipe navigation, lazy-batches candidates
  (initial 4, then batches of 3 once fewer than 2 remain ahead), and
  preserves playback continuity by rotating queue indices across three
  CSS-positioned `<video>` slots rather than re-parenting DOM nodes
- Statistics page (`?view=stats`) and Settings page (`?view=settings`),
  both reached from the Rail; Settings edits the annotation backend via
  `GET`/`PUT /api/settings`, probes a remote server with
  `POST /api/settings/annotation/{test,models}` (probe failures come back
  as `200` with `ok: false` so the browser console stays clean), and
  queues `POST /api/jobs/reannotate-all` behind the shared `ConfirmDialog`

## Legacy feature disposition

Retirement is not a parity project:

- **Retired:** the legacy-only bulk-select/delete-all UI, dedicated tags explorer,
  and live-connection panel/badge. Existing Atelier per-item deletion, tag chips,
  tag search, live counts and media-status updates remain.
- **Retained:** backend bulk-delete, tag, live-update and job-management APIs;
  shared authentication, root browser-cookie bootstrap and protected media serving.
- **Separate follow-up:** safe retry-failures controls and actionable health states
  belong to [UX-08](../meta/issues/clarify-library-health-and-error-recovery.md).
  No automatic port of the old retry-stuck/ops menu is promised here.

## Build pipeline

```
frontend/                                Source: Svelte 5 components, Tailwind v4
  src/App.svelte                         entry; effect-driven data loading
  src/components/                        Rail / Header / SearchBar / QuickRow / Masonry / Pin / Lightbox / Icon
  src/lib/api.ts                         typed fetch wrappers around /api/*
  src/lib/stores.ts                      svelte-store-based app state + URL sync
  src/lib/types.ts                       backend record + Pin types
  src/lib/utils.ts                       pin shaping, match-score, tag tone
  src/app.css                            @import "tailwindcss"; @theme tokens
  vite.config.ts                         dev proxy + outDir into Go embed root
  scripts/clean-dist.mjs                 remove old build artefacts but keep .gitkeep

internal/webui/atelier/dist/             Build output, embedded by Go
  .gitkeep                               sentinel so //go:embed always has a file
  index.html                             produced by Vite
  assets/                                produced by Vite
```

Vite is configured with `emptyOutDir: false` and a small pre-build node
script that wipes the previous `index.html` and `assets/`, so the
`.gitkeep` survives across builds and the Go embed always has at least one
file at compile time even on a fresh checkout.

The Go server (`internal/webui/http.go`) embeds the dist via
`//go:embed all:atelier/dist`. If `index.html` is missing (fresh clone, no
build) it serves a friendly placeholder explaining how to run the build.
This fallback exists only for source-only development; it has no alternate-shell
link. A plain `go build` is not a frontend build or a supported release recipe.

`npm run build` validates its output using `scripts/validate_atelier_build.mjs`.
The shared, dependency-free Node gate requires an app mount, emitted module JS
and CSS, and nonempty files for every `/assets/` reference in the index. It rejects
missing dist, the developer placeholder, the source index, and incomplete output.
It is a structural check; Playwright still verifies that the app actually runs.

Delivery paths:

- CI builds/checks Atelier **before** Go tests, so tests exercise freshly embedded
  assets, then runs the script regressions and (on Linux) browser smoke.
- `scripts/package_release.sh` builds and validates before Go compilation.
  `IMGSEARCH_SKIP_FRONTEND_BUILD=1` reuses a complete dist but never skips
  validation; Node 24+ is required even with that option.
- `Containerfile.cuda` builds Atelier in a Node 24 stage, validates it, then copies
  it into the Go build stage. Host dist and `node_modules` are excluded from the
  build context; Node is not present in the runtime image.
- `mise run build:frontend` and the normal `serve*` entry points build/validate.
  `mise run test:ui` builds first; direct `npm test` / `npm run test:atelier` require
  an existing complete build and **fail**, rather than skip, if it is absent.

## Dev workflow

```bash
# 1. Install deps once
mise run build:frontend            # or: cd frontend && npm install

# 2. In one terminal: the Go server (auto-restarts on .go and embedded asset changes)
mise run serve:8b:annotator-26b    # serves at :8081

# 3. In another terminal: the Atelier dev server with API proxy + HMR
mise run dev:frontend              # serves at :5173, proxies /api, /media to :8081
```

The Vite dev server reads `IMGSEARCH_API_KEY` from the repo-root `.env` and
forwards it on `/api/*`, `/media/*`, and `/healthz` requests, so the dev
session authenticates without manual cookie/header setup.

## Production-like testing

```bash
mise run build:frontend            # rebuilds dist
# Go autoreload picks up the new embedded assets within a few seconds.
# Open http://localhost:8081/ for Atelier.
```

## Annotation parallelism

In **Settings → Annotation → OpenAI-compatible**, set **Parallel annotations** and save. `1` runs one remote annotation job at a time; `2` permits two concurrent jobs when work is available. The existing persisted `annotation.openai.concurrency` value (1–64, default 2) now controls worker admission as well as the remote client's request limit.

Changes apply without restarting after the next settings check and any required drain. Checks use a five-second polling throttle; running embedding/transcription work can delay the next check. Lowering the limit does not cancel running work. Native inference remains serial; embedding/transcription work drains annotation jobs before running. Limits apply per worker process, not across multiple processes. Parallelism is capacity, not a promise of a 2× throughput improvement; provider limits and workload still matter.

## Tests

- `scripts/ui_smoke_test_atelier.mjs` boots an in-process API stub and exercises
  Atelier in Playwright, including library/search, tags, Feed, media actions,
  upload and settings. Run with `npm test`, `npm run test:ui` or
  `npm run test:atelier` after building; `mise run test:ui` builds first.
- `scripts/atelier_build_test.sh` covers invalid/missing dist, smoke failure,
  release skip-build validation and delivery wiring; included in `test:scripts`.
- `mise run check:frontend` runs `svelte-check` for type/lint feedback.
- `go test ./internal/webui ./internal/httputil` covers retired routes, developer
  fallback, emitted assets when built, root-cookie bootstrap, API/media auth,
  video MIME/ranges, and media directory/path protections. CI builds first.

## Decisions

- **SPA + Go embed** rather than SvelteKit: no Node runtime in production,
  one binary deploys, and we don't need SSR for a private library tool.
- **Query-string routing** (`?q=`, `?similar=`) rather than path-based:
  sidesteps the SPA history fallback, lets Go serve `index.html` only at
  exactly `/`, and keeps the URL form readable.
- **Tailwind v4 with `@theme`** for design tokens: utility classes like
  `bg-bg`, `text-ink`, `border-line` resolve into the Atelier palette
  without a separate JS config file.
- **One embedded shell:** Atelier is the only UI. Rollbacks mean restoring a
  previous binary, not following a legacy escape link. See the feature
  dispositions above; removal does not remove shared backend capabilities.
- **Upload as a modal** rather than a route: dragging files into the page
  feels lighter than a navigation, and a modal keeps the masonry context
  visible behind the dimmed backdrop. After a successful batch we bump
  `dataEpoch` so the library re-fetches in place rather than the user
  having to reload.
- **Feed via slot rotation, not DOM re-parenting**: the legacy feed
  physically `appendChild`'d the three `<video>` elements between
  `.feed-slide` parents on each swipe. In Svelte that fights the
  framework. The Atelier port keeps three persistent `<video>` elements
  and three "queue index" assignments; advancing the feed updates the
  assignments so each element's CSS slot (`translateY(-100% | 0 | 100%)`)
  changes without re-parenting. The element that was at "next" stays the
  same DOM node while it animates into the "current" slot, preserving
  its playing buffer.
- **Feed feedback is session-local**: tag preference scores live in a
  `Map` inside the component, decay 0.9 per fetch, clamp to the same
  `[-3, +5]` bounds as the legacy classifier, and disappear with the
  tab. Closing the overlay wipes everything. Batch requests send derived
  `prefer_tags` / `avoid_tags` CSVs and recent positive / soft-negative
  frame IDs so the backend can build a temporary adapted query vector;
  stored embeddings and user profiles are never mutated. Vector feedback uses
  a capped recent-ID window rather than the tag-score decay curve; the backend
  keeps its influence bounded with a normalized delta cap. Atelier also
  reranks already-buffered future items after tag feedback starting at
  `currentIndex + 2`, so the current and immediately preloaded next item stay
  stable.

## Navigation and search scope

Mobile uses labelled Library, Search, Collections and More destinations with 44px core targets; desktop keeps persistent labels. Utility views omit library search/tag chrome. Library-only media/sort filters are explicitly described rather than pretending the ranked-search API supports them. Browse tags replaces the query (not additive filtering), keeping current backend semantics explicit. Ranked results report a capped, non-exhaustive returned batch and relative Match scores, not confidence. One request key gates rendered cards/lightbox during scope changes; pagination retains same-scope results.

Settings drafts live only in a page-lifetime store, so internal navigation does not discard edits and entered credentials are not written to browser storage. Dirty drafts arm beforeunload; browsers may not deliver it during mobile process termination. Save/discard reset the baseline; worker-active status remains separate from unsaved inputs.

Lightbox keeps core viewing actions outside the metadata scroller, while edit mode shows Save changes/Cancel only. File details and advanced actions use disclosures. Closing or navigating still discards an unfinished Lightbox edit; Settings draft retention does not imply a global editor-draft policy.
