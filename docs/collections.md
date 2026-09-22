# Saved collections

Collections are named, persistent lists of existing images and videos. They belong to the same authenticated instance/library as the media—not to separate user accounts. One item may be saved to multiple collections.

Removing a membership or deleting a collection **never deletes the original media**. Deleting media from the library removes its memberships automatically. Collection names need not be unique; IDs identify collections.

## API

All routes use the existing `/api/` authentication and JSON error format (`{"error":"…"}`).

| Method | Route | Body / response |
| --- | --- | --- |
| GET | `/api/collections` | `{collections: [{id, name, created_at}]}`, newest collection first |
| POST | `/api/collections` | `{name}` → `201 {id, name, created_at}` |
| GET | `/api/collections/{id}` | `{id, name, created_at}` |
| PATCH | `/api/collections/{id}` | `{name}` → updated collection |
| DELETE | `/api/collections/{id}` | `204`; original media retained |
| GET | `/api/collections/{id}/items` | `{items, total}`, most recently saved first |
| PUT | `/api/collections/{id}/items/{image\|video}/{mediaID}` | No body; `204`, idempotent save |
| DELETE | `/api/collections/{id}/items/{image\|video}/{mediaID}` | `204`, idempotent removal |

- Names are trimmed and must contain 1–100 Unicode characters with no control characters. Create/rename accept only the `name` field.
- Use `GET /api/collections?media_type=image&media_id=123` (or `video`) to get the collections containing that item.
- Item pages accept `limit` (default 50, maximum 200), `offset`, and `include_nsfw` (default false). Invalid pagination parameters use the existing library fallbacks. NSFW filtering applies before pagination and counting, including tags on video frames.
- Each item is either `{media_type: "image", image: <existing ImageItem>}` or `{media_type: "video", video: <existing VideoItem>}`. Videos use their **video ID**, not a preview frame ID. Saving a derived frame separately returns `409`.
- Re-saving does not duplicate or reorder a membership. Remove then re-add to move it to the front.
- Missing collections/media return `404`; invalid IDs, media kinds, or names return `400`. Removing a nonexistent membership from an existing collection returns `204` even if its media was already deleted.

## Persistence and scope

Migration 16 adds `collections` and `collection_items`. Membership has exactly one image/video foreign key, unique per collection and target, with cascading cleanup. Media files, annotations, search indexes, and jobs are unchanged. Collection pages reuse existing image/video response hydration; the bounded per-item reads favor consistency with the library payload over a second copy of its SQL.

No sharing permissions, manual ordering, bulk saving, recurring imports, or legacy-shell collection UI are introduced.
