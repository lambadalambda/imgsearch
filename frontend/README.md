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
it on `/api/*`, `/media/*`, and `/legacy` requests so the browser session
authenticates automatically.

Open http://localhost:5173 — the legacy UI is still reachable at
http://localhost:8081/legacy.

## Build

```bash
cd frontend && npm run build
```

This builds into `internal/webui/atelier/dist/`, which is embedded by the Go
server at compile time. After building, run `go build ./...` (or any of the
`mise run serve*` tasks) and the new SPA is served at `/` with the legacy
shell at `/legacy`.

## Layout

- `src/App.svelte` — top-level app + data-loading effect
- `src/components/` — Rail / Header / SearchBar / QuickRow / Masonry / Pin / Lightbox / Icon
- `src/lib/` — `api.ts` (typed fetch wrappers), `stores.ts` (Svelte 5 stores),
  `types.ts`, `utils.ts`
- `src/app.css` — Tailwind v4 import + Atelier design tokens via `@theme`

The MVP scope covers library browse, text search, find similar, and a
lightbox. Feed mode, upload, bulk select/delete, NSFW flagging, and the tags
explorer still live in the legacy UI for now.
