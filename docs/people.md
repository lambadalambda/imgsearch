# People (face grouping)

People group faces across images and videos by identity, like Collections
group media by hand. Whole-image similarity search cannot do this: two photos
of the same person in different places have distant global embeddings.

## Data model (migration 17)

- `persons(id, uuid, name, created_at)`: a cluster of faces. `name`
  defaults to neutral `Person N`; the backend always keeps a UUID.
- `face_detections(id, image_id, person_id NULLABLE, box_*, embedding_blob,
  confidence, created_at)`: one detected face. Faces reference image rows
  only; video membership is derived through `video_frames`, so frame faces
  surface as their video (same rule as collections: save the video, not its
  derived frame). `person_id` is nullable so detector output can wait in an
  unassigned pool before clustering assigns it.

Deletion: deleting a person keeps its faces as unassigned (reassignable);
deleting media cascades to its faces. Membership removes nothing else.

## API

All routes use the existing `/api/` authentication and JSON error format.

| Method | Route | Notes |
| --- | --- | --- |
| GET | `/api/people` | `{people:[...]}`, newest first |
| GET | `/api/people?summary=true` | Adds `face_count` + up to 4 newest face-image covers |
| POST | `/api/people` | `{name}` optional; omitted yields neutral `Person N` auto-name, `201` |
| GET/PATCH/DELETE | `/api/people/{id}` | Rename/retire; names are 1–100 chars, no control chars |
| GET | `/api/people/{id}/items` | `{items,total,item_count}`, deduplicated media newest-face-first, NSFW-filtered before pagination like collections |
| POST | `/api/people/merge` | `{from_ids, into_id}`: moves faces, deletes sources |
| POST | `/api/people/{id}/split` | `{face_ids, name?}`: moves faces to a new person, `201` |
| GET | `/api/faces` | `?person_id=` or `?unassigned=true`, with thumbnails |
| POST | `/api/faces/{id}/reassign` | `{person_id}` or `{person_id:null}` |

Face creation belongs to the background detection worker (planned); the API
only clusters, corrects, and reads.

## Pipeline (done, behind `-enable-faces` + ONNX Runtime)

Detect (multi-face SCRFD-10G) → align (Umeyama similarity to the ArcFace
template) → identity embedding (ArcFace R50 `w600k_r50`, 512-d, L2
normalized) → greedy nearest-neighbor match to existing persons above cosine
0.5, else new `Person N`. Implemented in `internal/faces` (pure-Go
preprocessing, SCRFD decode, and alignment; `Engine` implements
`people.Detector` over ONNX Runtime) and run as `index_faces` jobs claimed
after embedding (`internal/worker`), enqueued on upload and for sampled
video frames when enabled, with `db.EnsureFaceJobsForModel` at startup and
for backfill. VLM annotators (Gemma/Qwen) stay on captions: they have no
stable identity metric.

Models: `det_10g.onnx` + `w600k_r50.onnx` from the InsightFace `buffalo_l`
release zip, downloaded and extracted on first run into
`./models/insightface/buffalo_l/` (`ensureDefaultFaceAssets`); custom paths
skip downloading. The engine reuses the Parakeet ONNX Runtime library
(`-faces-onnxruntime-lib`, falling back to `-parakeet-onnxruntime-lib`).
AVIF files have no pure-Go decoder and are skipped, not failed.
