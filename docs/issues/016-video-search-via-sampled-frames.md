# 016: Video Search via Sampled Representative Frames

## Status: Completed

## Goal

Add MVP video search by sampling a fixed number of representative frames from each uploaded video, embedding those frames with the existing image pipeline, and grouping frame hits back into video results during search.

## Scope Landed

- Upload endpoint now accepts supported video files in addition to images.
- Videos are stored in a new `videos` table and `videos/` storage directory.
- Each uploaded video is sampled into a fixed number of representative frames using `ffprobe` + `ffmpeg`.
- Sampled frames are stored as normal image assets, linked to videos through `video_frames`.
- Only `embed_image` jobs are created for sampled frames.
- Annotation jobs are intentionally skipped for sampled video frames.
- Search groups multiple frame hits for the same video into a single video result with:
  - `video_id`
  - video `storage_path`
  - best-match timestamp
  - preview path for the best matching frame
- Normal image uploads and image search behavior remain intact.
- Image gallery listing excludes derived video-frame images.

## Design Choices

### Fixed frame budget

The MVP uses a fixed frame count per video rather than time-based sampling.

- Default: `10` sampled frames per video
- Sampling strategy: uniform segment-center timestamps across the full duration

This keeps indexing cost bounded for long videos and avoids the storage explosion that would come from sampling at a fixed rate like 1 fps.

### Frame reuse through `images`

Sampled frames reuse the existing `images` table and image embedding/indexing pipeline.

That keeps the change small because:

- no new embedding type was required
- no vector-index schema changes were needed
- worker code could continue handling `embed_image` jobs only

The new `video_frames` association table links videos back to those sampled images.

### Search ranking

Video results are ranked by the best matching sampled frame.

- multiple frame hits for the same video collapse to one result
- the first/best hit determines the returned timestamp and preview frame

This is intentionally simple and should be revisited only if search quality proves insufficient.

## Known Limitations

- A fixed 10-frame budget can miss short but important moments in long videos.
- Search quality depends entirely on sampled visual frames; there is no transcript, OCR, audio, or scene-detection signal yet.
- Similar-search still operates on image IDs underneath; grouped video search is currently implemented in result enrichment rather than as a separate video-native index.
- Supported video formats are intentionally narrow and rely on the local `ffprobe` / `ffmpeg` toolchain.

## Follow-Up Ideas

- Scene-aware or keyframe-aware sampling instead of pure uniform sampling.
- Optional transcript/OCR fusion for richer text-to-video retrieval.
- Dedicated `/api/videos` listing and browsing endpoints.
- Better ranking for videos using the top N frame scores rather than just the best frame.
