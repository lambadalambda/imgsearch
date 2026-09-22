# Saved collections

Collections are named, persistent lists of existing images and videos. They belong to the same authenticated instance/library as the media—not to separate user accounts. One item may be saved to multiple collections.

Removing a membership or deleting a collection **never deletes the original media**. Deleting media from the library removes its memberships automatically. Collection names need not be unique; IDs identify collections.

## In Atelier

- Open **Collections** in the navigation to create and browse lists, or rename/delete an existing list.
- Use **Save to collection** on an image/video card, in the lightbox, or in the video Feed. Select one or more existing lists, or create a list while saving.
- Reopen the picker to remove a saved membership. This differs from the media's **Delete** action, which deletes it from the library.
- Collections can mix images and videos and display the newest saves first. Your NSFW visibility preference still applies; hidden media remains saved.
- Collection URLs (`?view=collections` and `?view=collection&collection_id=123`) can be bookmarked and revisited after restarting the app.

## API

All routes use the existing `/api/` authentication and JSON error format (`{"error":"…"}`).

| Method | Route | Body / response |
| --- | --- | --- |
| GET | `/api/collections` | `{collections: [{id, name, created_at}]}`, newest collection first |
| GET | `/api/collections?summary=true&include_nsfw=false` | `{collections: [{id, name, created_at, item_count, visible_count, covers}]}`; lightweight card summaries |
| POST | `/api/collections` | `{name}` → `201 {id, name, created_at}` |
| GET | `/api/collections/{id}` | `{id, name, created_at}` |
| PATCH | `/api/collections/{id}` | `{name}` → updated collection |
| DELETE | `/api/collections/{id}` | `204`; original media retained |
| GET | `/api/collections/{id}/items` | `{items, total, item_count}`, most recently saved first; `total` is visible, `item_count` is all saved |
| PUT | `/api/collections/{id}/items/{image\|video}/{mediaID}` | No body; `204`, idempotent save |
| DELETE | `/api/collections/{id}/items/{image\|video}/{mediaID}` | `204`, idempotent removal |

- Summary mode is opt-in (`summary=true`) and defaults to hiding NSFW content. Plain listing, picker membership lookups and CRUD responses keep their original lightweight shape. Membership-target queries remain plain even if `summary=true` is supplied.
- `item_count` includes hidden memberships; `visible_count` counts those allowed by the request's NSFW preference. `covers` is an array of at most four newest visible saves, each `{media_type, media_id, thumbnail_path?}`. Filtering happens before cover selection, including NSFW tags on any video frame; hidden IDs/paths are never returned as covers. Counts intentionally reveal that saved items exist, within the existing authenticated shared-library model.
- Covers use image/frame derivative thumbnails; video previews use the earliest frame index. A frameless video has no thumbnail path and should render a neutral tile. One set-based summary query avoids full media hydration per index card; counts/ranking still process memberships, so query count is fixed but total work is not constant. No schema change or persisted counters were added.
- Names are trimmed and must contain 1–100 Unicode characters with no control characters. Create/rename accept only the `name` field.
- Use `GET /api/collections?media_type=image&media_id=123` (or `video`) to get the collections containing that item.
- Item pages accept `limit` (default 50, maximum 200), `offset`, and `include_nsfw` (default false). Invalid pagination parameters use the existing library fallbacks. NSFW filtering applies before pagination and counting, including tags on video frames.
- Each item is either `{media_type: "image", image: <existing ImageItem>}` or `{media_type: "video", video: <existing VideoItem>}`. Videos use their **video ID**, not a preview frame ID. Saving a derived frame separately returns `409`.
- Re-saving does not duplicate or reorder a membership. Remove then re-add to move it to the front.
- Missing collections/media return `404`; invalid IDs, media kinds, or names return `400`. Removing a nonexistent membership from an existing collection returns `204` even if its media was already deleted.

## Persistence and scope

Migration 16 adds `collections` and `collection_items`. Membership has exactly one image/video foreign key, unique per collection and target, with cascading cleanup. Media files, annotations, search indexes, and jobs are unchanged. Collection pages reuse existing image/video response hydration; the bounded per-item reads favor consistency with the library payload over a second copy of its SQL.

No sharing permissions, manual ordering, bulk saving, recurring imports, or legacy-shell collection UI are introduced.
