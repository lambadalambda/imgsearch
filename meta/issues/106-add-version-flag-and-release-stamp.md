# Add a `-version` flag and stamp release builds

## Summary

There is no `-version` flag among the 46 flags in `cmd/imgsearch/runtime_config.go`, `package_release.sh:146` builds with `-ldflags='-s -w'` and no `-X` stamp, and the single mutable `rolling` tag replaces its artifacts on every push. Users cannot tell which commit their binary came from, and `CHANGELOG.md` has a 160-line Unreleased section with date headings instead of versions.

## Requirements

- Add `-version` printing version, commit SHA, and build date, populated via `-ldflags -X`.
- Stamp the rolling tarball filename with the short SHA and include the version in the release `README.txt`.
- Expose the version in `/healthz` or `/api/stats` so the UI can show it in the statistics pane.
- Decide on a lightweight versioning convention (for example date-based tags cut from Unreleased) and document it in `docs/development.md`.

## Acceptance Criteria

- `./imgsearch -version` prints a non-empty commit SHA in a release build.
- Release artifact names include the SHA.

## Notes

- Found during the 2026-09-21 review.
