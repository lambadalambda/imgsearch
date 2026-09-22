# Import recent media from Fediverse accounts

## Summary

Add a command-line script that imports a Fediverse account's public image/video attachments posted in the last N days, starting with `https://stereophonic.space/users/genmaicha`.

## Requirements

- Accept a profile URL, configurable day window, and imgsearch destination/API-key environment settings.
- Resolve accounts and paginate their posts on Mastodon-compatible servers (including Pleroma/Akkoma).
- Import original attachments, not previews; exclude boosts and posts older than the cutoff.
- Reuse the existing upload/conversion path where practical; handle duplicates, rate limits, and individual failures clearly.
- Offer a read-only preview/dry-run and document limitations (publicly accessible media; unsupported audio skipped).
- Never forward imgsearch credentials to a Fediverse server or media host.

## Acceptance Criteria

- Tests cover pagination, cutoff boundaries, attachment selection/deduplication, failure reporting, and credential separation.
- A documented example accepts the supplied profile URL and a day window.
- Verify discovery against the supplied server without performing a bulk production import.

## Notes

- Scope is a user-invoked importer, not recurring sync, account authentication, or a server-side UI feature.

## Outcome

- Implemented `scripts/import_fediverse.py` and `mise run import-fediverse`, reusing the existing Bash upload/conversion helper. Documented usage and public Mastodon-compatible API limitations in README/development docs.
- All 14 offline tests pass, covering discovery/pagination/date boundaries, original attachment selection, credential separation, real CLI-to-uploader integration, bounded/truncated downloads, and stopping on exhausted rate limits or long requested backoff. Initial implementation and follow-up regressions followed red → green; independent review approved the final code.
- Read-only discovery of the supplied account with `--days 1 --dry-run` selected 10 attachments successfully. No production uploads performed.
- Final `mise run test:full` passed Go, all 31 frontend unit tests, script regressions, and both browser smoke suites. Formatting and whitespace checks passed. Earlier full/UI runs hit an intermittent Atelier visual search-rank assertion, tracked separately in [the smoke failure issue](investigate-atelier-visual-search-rank-smoke-failure.md); no frontend changes made here.
