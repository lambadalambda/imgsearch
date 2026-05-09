# Redesign mockups

Three exploratory directions for the imgsearch web UI. Each is a single
self-contained HTML file in this directory:

- `atlas.html` — dark, minimal, command-bar-centric (Linear / Raycast / Mubi)
- `atelier.html` — warm, masonry, mood-board (Are.na / Pinterest / Cosmos)
- `studio.html` — three-pane, dense, pro tool (Lightroom / Bridge / Finder)

Open `index.html` for a side-by-side gallery.

## How to view

```bash
open docs/mockups/index.html
```

The mockups load placeholder thumbnails from `picsum.photos` so they need a
network connection on first load. They contain no JavaScript and no fetches
to the live imgsearch server — they are pure layout/typography studies.

## Direction summary

| | Atlas | Atelier | Studio |
| --- | --- | --- | --- |
| Mood | Dark, quiet, technical | Bright, warm, editorial | Pro tool, dense, neutral |
| Primary metaphor | Search box, library is a backdrop | Mood board / collection | Media library with inspector |
| Density | Low (image-first) | Medium (image + tags) | High (metadata always visible) |
| Tabs / nav | Inline facet chips below cmd | Icon rail + saved collections | Sidebar tree + saved searches |
| Cards | Bare thumbs, hover reveals meta | Pinned tiles with title + tags | Tight thumbs, selection-driven |
| Selection model | Single-target → side drawer | Save / open per card | Multi-select with batch tools |
| Mobile | Same cmd bar, full-bleed grid | Rail collapses to top dock | Inspector becomes bottom sheet |
| Best fit | "this is a search tool" | "this is my mood board" | "this is my media library" |

## Picking a direction

Things to weigh while comparing:

- How much do users come here to **search** vs **browse**?
- How important is bulk/selection workflow (delete, tag, re-annotate)?
- Should the tool feel personal/warm or technical/neutral?
- Is dark or light the primary mode (or both)?
- How much metadata do power users want visible at all times?

If two directions are tied on feel, the IA differences are usually the
deciding factor — Atlas optimises for one query at a time, Atelier optimises
for browsing a collection, Studio optimises for working through a selection.

## Iterating

Each file is intentionally small (~400 lines including embedded CSS) and uses
no build step. Tweak palettes via the `:root` custom properties at the top.
Once a direction is picked, the next step is porting it into
`internal/webui/static/{styles.css,index.html}` incrementally, behind the
existing smoke test coverage in `scripts/ui_smoke_test.mjs`.
