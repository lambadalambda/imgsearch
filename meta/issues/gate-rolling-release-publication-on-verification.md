# Gate rolling release publication on verification

## Summary

The rolling-release workflow runs independently of CI on every main/master push. Its publish job depends only on packaging, so a commit can replace the rolling download even if its CI tests fail.

## Requirements

- Require successful verification of the same commit SHA before publishing the rolling release, including manual dispatches.
- Reuse the existing CI checks rather than maintaining a second divergent test suite.
- Keep packaging parallel where practical; gate publication rather than requiring unnecessary serial builds.

## Acceptance Criteria

- A failed verification run cannot replace the rolling release.
- A successful verification and packaging run can publish artifacts for the verified SHA.
- Manual dispatch cannot bypass verification or publish artifacts for a different SHA.

## Notes

- Evidence: `.github/workflows/ci.yml` and `.github/workflows/rolling-release.yml`; the latter's `release` job has only `needs: build`.
- A reusable verification workflow is one possible approach; choose the smallest maintainable solution.
