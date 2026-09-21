# Fix documentation drift

## Summary

Several docs describe a previous state of the project:

- `AGENTS.md:7` says to use plain HTML/CSS/JavaScript; the frontend is Svelte 5 + Tailwind 4 + Vite.
- `docs/development.md:63` references `github.com/cshum/vipsgen`, which was replaced by direct libvips calls and is no longer in `go.mod`. Its "UI Summary" describes the legacy shell, and it never mentions `build:frontend` or `dev:frontend`, so a new contributor following it gets the "Atelier frontend not built yet" page.
- `docs/architecture.md` is titled "Planned Architecture", describes a `./data/thumbs/` directory that does not exist, and its data model omits videos, frames, transcripts, and annotations.
- `README.md` Notes says supported formats are JPEG, PNG, WEBP, and AVIF; uploads also accept MP4, WebM, QuickTime, and Matroska, and GIF import is tested. Quick Start tells all platforms to install libvips and run `./imgsearch`, but Linux tarballs bundle libvips and ship `run.sh`.
- `docs/decisions.md` has no ADR for the Svelte migration, video support, or the annotation model choice.
- `docs/screenshot.png` (3.2 MB) is tracked but unreferenced; only `screenshot.webp` is used.

## Requirements

- Correct each item above.
- Have `mise run serve` depend on `build:frontend`, or document the step prominently.
- Remove `docs/screenshot.png`.

## Acceptance Criteria

- A new contributor can follow `docs/development.md` from clone to a working Atelier UI without consulting other files.
- No doc references vipsgen, a thumbs directory, or a plain-JS frontend.

## Notes

- Found during the 2026-09-21 review.
