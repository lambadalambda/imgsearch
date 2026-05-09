# Frontend architecture

The web UI is in transition. Two shells coexist:

- **Atelier** at `/` — the new Svelte 5 + Tailwind v4 + Vite SPA, source in
  `frontend/`, build output embedded into the Go binary at
  `internal/webui/atelier/dist/`.
- **Legacy** at `/legacy` — the original `internal/webui/static/`
  vanilla-JS shell. Kept as a fallback for unported features (bulk
  select/delete, tags explorer, ops menu, live status) and for
  one-click rollback if the new SPA misbehaves.

Both shells share the same backend API at `/api/*` and media serving at
`/media/*`.

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

The following still live in `/legacy` for now and will be ported in
follow-ups:

- Bulk select / delete
- Tags explorer page
- Ops menu (retry stuck, refresh, queue stats)
- Live update connection badge

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
forwards it on `/api/*`, `/media/*`, and `/legacy` requests, so the dev
session authenticates without manual cookie/header setup.

## Production-like testing

```bash
mise run build:frontend            # rebuilds dist
# Go autoreload picks up the new embedded assets within a few seconds.
# Open http://localhost:8081/ for Atelier; http://localhost:8081/legacy for the legacy UI.
```

## Tests

- `scripts/ui_smoke_test.mjs` — legacy SPA smoke tests, now pointed at
  `/legacy` and serving the legacy index/assets in the same in-process node
  server.
- `scripts/ui_smoke_test_atelier.mjs` — minimal Atelier smoke test that
  boots an in-process API stub and walks: library renders → text search →
  find similar → lightbox open/close. Skips with success when
  `internal/webui/atelier/dist/index.html` is missing so a fresh clone does
  not fail before the first frontend build.
- `npm test` runs both. Individual: `npm run test:legacy`,
  `npm run test:atelier`.
- `mise run check:frontend` runs `svelte-check` for type/lint feedback on
  the SPA.
- `go test ./internal/webui` covers the Atelier/legacy routing and embed.

## Decisions

- **SPA + Go embed** rather than SvelteKit: no Node runtime in production,
  one binary deploys, and we don't need SSR for a private library tool.
- **Query-string routing** (`?q=`, `?similar=`) rather than path-based:
  sidesteps the SPA history fallback, lets Go serve `index.html` only at
  exactly `/`, and keeps the URL form readable.
- **Tailwind v4 with `@theme`** for design tokens: utility classes like
  `bg-bg`, `text-ink`, `border-line` resolve into the Atelier palette
  without a separate JS config file.
- **Two embed declarations** rather than versioned URL prefixes: legacy and
  Atelier ship together in the same Go binary so we can roll back to the
  legacy UI by visiting `/legacy` without a redeploy.
- **MVP first**: the SPA does *not* yet replicate bulk select, the tags
  explorer, the ops menu, or the live-update connection badge. Each of
  those is a separate follow-up issue.
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
  tab. Closing the overlay wipes everything. The only thing that ever
  reaches the server is the `prefer_tags` / `avoid_tags` CSV.
