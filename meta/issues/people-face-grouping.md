# People / face grouping

Group faces across images and videos by identity (People view, like
Collections but automatic), with user rename/merge/split/reassign.

Decisions (2026-11-21): images + sampled video frames; auto-cluster on
ingest with neutral `Person N` auto-names + backend UUID; local ONNX
detector + identity embedding (existing `onnxruntime_go`, no new services);
VLM annotators stay on captions and are never used for identity. See
`docs/people.md`.

## Scope

- [x] Migration 17: `persons` + `face_detections` (image-anchored, nullable
      `person_id`, delete-person-keeps-faces, delete-media-cascades).
- [x] People CRUD + auto-naming + `?summary=true` with covers.
- [x] `GET /api/people/{id}/items`: deduplicated media, newest-face-first,
      NSFW-filtered before pagination; frame faces surface as videos.
- [x] Corrections: merge, split, face reassign/unassign, unassigned pool.
- [x] Worker: face detection + identity embedding + auto-cluster job kind
      (`index_faces`, `people.Detector` interface, greedy centroid match in
      `people.AssignDetections`), enqueue on upload (`EnableFaces`), backfill
      helper (`db.EnsureFaceJobsForModel`), retry allowlist + capability.
- [x] Local ONNX models: SCRFD-10G detector + ArcFace R50 recognition
      (`internal/faces`, validated end-to-end on real photos), InsightFace
      `buffalo_l` zip download + extract on first run, `-enable-faces` and
      `-faces-*` flags with Parakeet library fallback, startup backfill call,
      AVIF skip (no pure-Go decoder).
- [x] Atelier UI: `?view=people` grid (create/rename/delete/merge, covers),
      `?view=person` detail reusing the results grid, face management
      (select, move/unassign, split into new person).
- [ ] Atelier UI: `?view=people` grid, person detail, rename/merge/split/
      reassign, face-box toggle in lightbox, multi-face badges.
- [ ] Docs/changelog follow-ups per commit.

## Risks

- Wrong auto-merges: mitigate with low clustering threshold + visible
  correction actions (merge/split/reassign shipped before the worker).
- Face jobs multiply video work: only sampled frames, never every frame.
- Embedding-table growth: face vectors stay out of `sqlite-vector` until
  scale demands it (brute-force cosine is fine for tens of thousands).
- Privacy: fully local, no third-party calls; document the guarantee.

## Acceptance

- Import a folder with repeated faces: each identity appears once under
  People with a neutral name; renaming shows all their photos/videos.
- Multi-face photos list every person in it.
- NSFW filtering matches collections behavior; deleting media/person never
  deletes other media.
