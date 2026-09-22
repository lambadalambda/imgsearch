# Turn operational states into next steps

## Priority

P2 — Design recommendation.

## Summary

Scheduler vocabulary and generic errors tell users something happened, but not what it means or how to recover.

## Evidence

Production Statistics shows queued and ready counts together, plus 468 videos needing a transcription job with none queued. It does not explain whether transcription is unavailable or simply incomplete. An injected local search-network failure displayed Failed to fetch twice, without Retry.

Reproduction / inspection steps:

1. Open Statistics and compare the queued/ready/processed labels and denominators.
2. In a disposable environment, interrupt a search request and inspect the resulting message.

## Risks

“Ready” sounds completed to a non-operator. An empty progress bar without an explanation can look broken even when a feature is intentionally disabled. Raw fetch errors do not offer a recovery path.

## Requirements

- Use Waiting / Processing / Complete / Failed as primary states; explain scheduler subsets under Advanced details.
- Show Not configured / Disabled / No audio when applicable, rather than implying every video has missing work.
- Label totals that include sampled video frames, and distinguish them from standalone media counts.
- Use one plain-language error with Retry preserving the query/filters. Offer a path from failures to the appropriate recovery action.

## Acceptance Criteria

- [ ] Every stalled/disabled/failed state explains its scope and a useful next step; recoverable errors offer Retry without losing context.
- [ ] Add red→green coverage for behavior changes; verify desktop and mobile UI.

## Notes

- Audit only; no fix implemented.
- [HTML report, UX-08](../../docs/reviews/2026-09-22-atelier-ux-audit.html#UX-08) contains embedded screenshot evidence.
