# Build libvips with EXIF support in CI and Linux releases

## Summary

The shared `native-deps` GitHub action builds libvips 8.18 from source without `libexif-dev`. Linux CI therefore fails `TestThumbnailWebP/rotated` (first run on GitHub on 2026-10-14: a 400×640 EXIF-rotated image produced a 640×400 thumbnail). Linux release archives bundle the same build, so EXIF-rotated photos get sideways thumbnails and are likely embedded sideways too. Production on CT107 uses Ubuntu's packaged libvips and is not affected.

## Requirements

- Install `libexif-dev` before the libvips build and invalidate the cached libvips build.
- Fail the build early if libvips lacks EXIF support.

## Acceptance Criteria

- Ubuntu CI passes, including `TestThumbnailWebP/rotated`.
- The next rolling release Linux archive is built from a libvips with EXIF support.

## Notes

- The rolling release does not wait for CI; see [gate rolling release publication](gate-rolling-release-publication-on-verification.md).
