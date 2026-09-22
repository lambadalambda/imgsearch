# imgsearch frontend

Atelier UI for imgsearch — Svelte 5 + Tailwind v4 + Vite + TypeScript.

## Develop

```bash
# install once
cd frontend && npm install

# in one terminal: run the Go server
mise run serve:8b:annotator-26b   # or any serve task

# in another terminal: run Vite with API proxy
cd frontend && npm run dev
```

The dev server reads `IMGSEARCH_API_KEY` from the repo-root `.env` and forwards
it on `/api/*`, `/media/*`, and `/healthz` requests so the browser session
authenticates automatically.

Open http://localhost:5173 (development) or http://localhost:8081/ (embedded Atelier).

## Build

```bash
cd frontend && npm run build
```

This builds into `internal/webui/atelier/dist/`, which is embedded by the Go
server at compile time. After building, run `go build ./...` (or any of the
`mise run serve*` tasks) and Atelier is served at `/`. The build validates the
index and emitted JS/CSS before succeeding. Retired `/legacy` URLs return 404.

## Layout

- `src/App.svelte` — top-level app + data-loading effect
- `src/components/` — Rail / Header / SearchBar / QuickRow / Masonry / Pin / Lightbox / Icon
- `src/lib/` — `api.ts` (typed fetch wrappers), `stores.ts` (Svelte 5 stores),
  `types.ts`, `utils.ts`
- `src/app.css` — Tailwind v4 import + Atelier design tokens via `@theme`

Atelier supports library/search, lightbox and media actions, Feed, upload,
settings and statistics. Legacy-only bulk-delete UI, tags explorer and the
live-connection panel are retired, not pending parity work. See
[frontend architecture](../docs/frontend.md) for feature dispositions and delivery
gates. Browser smoke (`npm test` at the repo root) requires built assets and fails
on missing/incomplete dist; `mise run test:ui` builds them first.
