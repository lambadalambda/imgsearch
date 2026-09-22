# Atelier UI/UX audit — 2026-09-22

Standalone report: [2026-09-22-atelier-ux-audit.html](2026-09-22-atelier-ux-audit.html)

10 findings; 3 P1, 6 P2, 1 P3. Three reproduced workflow issues, seven recommendations. 23 embedded screenshot panels.

Private artifact: contains production media, filenames, hostnames and configuration details. Redact before external distribution. Priority describes impact; shipping order also accounts for small low-risk fixes. Effort is for the core change, with related improvements separate.

## UX-01 · P1 · Cleanup should preserve the places an image was saved
Reproduced workflow issue · Duplicates × Collections · Effort M

In the disposable sandbox, Audit scratchpad contained Cat duplicate variant B. Keeping the largest image in that duplicate group left Audit scratchpad empty. Repeated with a second group (Square companion / Dog portrait) with the same result. The confirmation names dimensions and deletion count, but does not mention affected collections.

### Suggested change
- Make Keep largest a consolidation operation: union collection memberships onto the keeper before deleting copies, in one transaction.
- Related improvement: preserve manual metadata deliberately; surface conflicts instead of silently choosing a winner.
- Preview the keeper, number of removals, and affected collections. State whether files will be deleted or retained.

### Reproduce / inspect
1. Save a smaller duplicate into a collection; confirm it appears there.
2. Open Duplicates, choose Keep largest for that group, and confirm deletion.
3. Return to the collection: the saved image is gone and the surviving copy is not in its place.

Acceptance: A collection containing any removed duplicate still contains the keeper after cleanup, with deterministic metadata handling and rollback on failure.
Issue: `meta/issues/preserve-collections-during-manual-duplicate-cleanup.md`
- Screenshot: 20-collection-before-cleanup.png — Before: the smaller copy is saved in Audit scratchpad. (Sandbox)
- Screenshot: 21-keep-largest-confirm.png — The dialog warns about deletion, but not collection loss. (Sandbox)
- Screenshot: 22-collection-after-cleanup.png — After: the collection is empty. Reproduced in two groups. (Sandbox)

## UX-02 · P1 · Clarify what “Exact only” means
Design recommendation · Duplicates · Effort S–M

The production view offers Exact only and labels groups as copies. In the first captured group, all five images are 450×424, but only the first is labelled largest. There is no visible explanation of ties or quality. The established feature contract is dHash equality/distance—not byte or pixel equality.

### Suggested change
- Rename Exact only to Identical visual hash; call the groups Possible matches. Explain that colors, edits, and fine detail may differ.
- Show an explicit Keep badge and let the user select another keeper. For equal dimensions say Same resolution, rather than implying one is larger.
- Add file size/format and an accessible side-by-side comparison. On phones, offer a two-up/flip comparison instead of only a long vertical list.
- Show the separate unattended-import policy/status when that feature is exposed in UI; do not imply the new import option fixes manual cleanup.

### Reproduce / inspect
1. Open Duplicates and select Exact only.
2. Inspect a group with equal dimensions and the keeper label.

Acceptance: No label claims byte equality without verification; users can identify the keeper and understand the deletion/retention policy before confirming.
Issue: `meta/issues/clarify-perceptual-duplicate-safety-and-keeper-choice.md`
- Screenshot: 05-duplicates-exact.png — Production: “Exact only”, “5 copies”, equal dimensions, one “largest” label. (Production)
- Screenshot: 34b-duplicates-mobile-loaded.png — Mobile comparison becomes a vertically stacked sequence. (Production · 390px)

## UX-03 · P1 · Give mobile navigation fewer jobs
Design recommendation · Navigation & page layout · Effort M

At 390×844, the first media begins around y=373—44% of the viewport. Navigation buttons measure about 29×40 CSS pixels. Library and Collections use near-identical grid icons; disabled Tags and the legacy link consume space. Settings/Statistics/Collections also inherit the full search-and-tag band.

### Suggested change
- Use Library, Search, Collections and More as the primary mobile navigation; move Statistics, Duplicates and Settings under More.
- Use persistent desktop labels and a distinct Collections icon. Remove Tags (soon) until useful and retire the legacy escape link.
- Collapse media/sort/NSFW options behind one clearly labelled filter control on small screens; retain a visible active-filter summary.
- Use a compact utility-page header instead of repeating global tags/counts. Aim for comfortable 44px touch targets; this audit is not a formal WCAG conformance test.

### Reproduce / inspect
1. Open the library at 390×844 and compare the space above the first image.
2. Open Settings at the same width; note the same search/tag band above the form.

Acceptance: Proposed target to agree during implementation: labelled core navigation with 44px touch targets and first media within the top third of a 390×844 viewport, without losing filter discoverability.
Issue: `meta/issues/simplify-atelier-navigation-and-mobile-chrome.md`
- Screenshot: 11-library-mobile.png — Mobile library: navigation, heading, filters, search, chips, then results. (Production · 390px)
- Screenshot: 13c-settings-mobile.png — The same browsing chrome pushes Settings controls down the page. (Production · 390px)
- Screenshot: 01-library-desktop.png — Keep the calm visual identity; improve navigation clarity rather than redesigning everything. (Production · 1440px)

## UX-04 · P2 · Make search modes and scope explicit
Design recommendation · Search, tags & counts · Effort M

Twice in production: search “a quiet forest at dawn” → 48 results → click Tag · outdoor → the query clears, heading becomes outdoor, and URL becomes ?tag=outdoor. Media/sort controls disappear in search. Results show bare 42%/39% badges. The library header's image/video totals and visible-result count have different, unexplained scopes.

### Suggested change
- Choose a clear model: additive tag filters with removable chips, or explicitly labelled Browse tag shortcuts separate from the search controls.
- Preserve the query and filter scope visibly; make media filtering consistent between browsing and searching where supported.
- Label similarity as a relative match score, or replace the raw percentage with rank/relevance presentation. Do not present it as confidence.
- Distinguish Library totals from Visible results; use media/image/card consistently instead of pins and Quick collections for tag shortcuts.
- Provide a visible Search by image button that opens a picker, especially for mobile and keyboard users; paste/drop can remain shortcuts.

### Reproduce / inspect
1. Search for a quiet forest at dawn.
2. Click Tag · outdoor under the search box.
3. Observe that this replaces the query rather than refining it.

Acceptance: Users can predict whether a tag click refines or replaces a search; counts and match indicators identify their meaning and scope.
Issue: `meta/issues/clarify-search-filter-scope-and-match-labels.md`
- Screenshot: 09c-search-before-tag.png — Before: a text query and ranked results. (Production)
- Screenshot: 10c-search-after-tag.png — After clicking the chip: query cleared, tag browsing instead. (Production)

## UX-05 · P2 · Protect unsaved Settings edits
Reproduced workflow issue · Settings · Effort S–M

In the sandbox, changed the native variant from e4b to 26b without saving, navigated to Library, then back to Settings: e4b returned and Save was disabled. Repeated via Collections. There was no discard warning or retained draft. No model configuration was saved.

### Suggested change
- Add a dirty-state guard with Stay / Discard changes, or retain an explicit unsaved draft across internal navigation.
- Keep a Save changes / Discard bar near edited controls, and show a clear Saved confirmation.
- Related layout improvement: avoid equal-height Settings cards that create a large empty backend panel; use independent panels or a compact form layout.

### Reproduce / inspect
1. Settings → change a field; do not save.
2. Choose Library or Collections, then return to Settings.
3. The prior value is restored without warning.

Acceptance: Leaving a dirty form cannot silently erase edits; the user can distinguish draft, saved, and active worker configuration.
Issue: `meta/issues/protect-unsaved-atelier-settings-edits.md`
- Screenshot: 23-settings-unsaved.png — Unsaved 26b selection, with Save enabled. (Sandbox)
- Screenshot: 24-settings-draft-lost.png — Returning to Settings restores e4b; the draft is gone. (Sandbox)

## UX-06 · P2 · Separate viewing actions from editing actions
Design recommendation · Lightbox & metadata · Effort S–M

The desktop editor shows Save to collection beside Save, then Cancel / Find similar / Open original / Flag NSFW / Re-annotate / Delete. On mobile, a long hash filename and description push these actions below the visible sheet. Actual title/tag editing and saving succeeded in the sandbox.

### Suggested change
- During editing, emphasize Save changes and Cancel; move unrelated actions out of the edit flow.
- In viewing mode, prioritize Save to collection, Similar, and More; keep destructive/reprocessing actions in the latter.
- Make a compact action row sticky within the mobile sheet and collapse filename/technical fields under File details.
- Retain the current title/summary/description hierarchy and previous/next navigation—they are useful.

### Reproduce / inspect
1. Open an image and choose Edit.
2. Compare the action area with the phone lightbox's initial viewport.

Acceptance: Save changes is unambiguous and reachable; core viewing actions remain easy to find without reading the entire description.
Issue: `meta/issues/simplify-lightbox-actions-and-edit-mode.md`
- Screenshot: 29-edit-desktop.png — Two save concepts and unrelated actions share the editing surface. (Sandbox)
- Screenshot: 12-lightbox-mobile.png — Mobile: title, hash filename and prose precede the actions. (Production · 390px)

## UX-07 · P2 · Make Collections feel like saved media
Design recommendation · Collections · Effort M

Populated and empty collections look alike in the list. Rename/Delete occupy much of each row. The empty-detail state says items might be NSFW-hidden even for a genuinely empty list, and offers no direct Browse library action. Create, save, rename and delete worked in the sandbox.

### Suggested change
- Add a small cover mosaic and item count; make the whole card a clearly clickable browse target.
- Move Rename/Delete into a labelled overflow menu while keeping delete confirmation and the promise that originals are retained.
- Distinguish No saved items yet from Items hidden by filters, with Browse library or Show hidden items as appropriate.
- Keep saved ordering stable and show short confirmation when a picker adds/removes an item.

### Reproduce / inspect
1. Open Collections with one populated and one empty list.
2. Open the empty list and inspect its next-step guidance.

Acceptance: A user can distinguish populated/empty/filtered collections at a glance and reach the next useful action directly.
Issue: `meta/issues/improve-collection-browsing-and-empty-states.md`
- Screenshot: 15-collections-populated.png — A populated list and an empty list have the same visual treatment. (Sandbox)
- Screenshot: 17-collection-empty.png — The empty state explains the feature, but leaves the next action implicit. (Sandbox)

## UX-08 · P2 · Turn operational states into next steps
Design recommendation · Statistics & error recovery · Effort S–M

Production Statistics shows queued and ready counts together, plus 468 videos needing a transcription job with none queued. It does not explain whether transcription is unavailable or simply incomplete. An injected local search-network failure displayed Failed to fetch twice, without Retry.

### Suggested change
- Use Waiting / Processing / Complete / Failed as primary states; explain scheduler subsets under Advanced details.
- Show Not configured / Disabled / No audio when applicable, rather than implying every video has missing work.
- Label totals that include sampled video frames, and distinguish them from standalone media counts.
- Use one plain-language error with Retry preserving the query/filters. Offer a path from failures to the appropriate recovery action.

### Reproduce / inspect
1. Open Statistics and compare the queued/ready/processed labels and denominators.
2. In a disposable environment, interrupt a search request and inspect the resulting message.

Acceptance: Every stalled/disabled/failed state explains its scope and a useful next step; recoverable errors offer Retry without losing context.
Issue: `meta/issues/clarify-library-health-and-error-recovery.md`
- Screenshot: 02-statistics-desktop.png — Statistics is informative, but scheduler terms and missing-job states need interpretation. (Production)
- Screenshot: 33-search-network-error.png — Injected network failure: duplicated technical wording, no recovery control. (Sandbox · injected failure)

## UX-09 · P3 · Validate uploads early; make failures recoverable
Design recommendation · Upload · Effort S

Browser file-input automation supplied a .txt file alongside a valid WebM. Both entered the pending queue. The server then rejected the .txt and accepted the video, with clear per-file outcomes. This bypassed the native chooser's accept filter; it is not proof the normal chooser lists unsupported files.

### Suggested change
- Reuse one client-side MIME/extension preflight for drop and picker inputs, while keeping server validation authoritative.
- Explain rejected files before upload, without adding them to a ready-looking queue.
- Offer Retry failed for transient failures and a correction/remove action for unsupported formats; preserve successful rows.
- Use one vocabulary for Upload / Add media / Import and explain duplicate outcomes separately from new uploads.

### Reproduce / inspect
1. In the sandbox, queue an unsupported file and a valid video through browser file-input automation.
2. Submit and inspect the partial-success summary and available follow-up controls.

Acceptance: Unsupported drops receive an immediate explanation; a transient failed row can be retried without resubmitting successful files.
Issue: `meta/issues/063-filter-upload-drop-files-client-side.md`
- Screenshot: 27-upload-invalid-pending.png — Both files appear ready to upload before validation. (Sandbox)
- Screenshot: 28-upload-partial-result.png — Clear partial success is a strength; add targeted recovery. (Sandbox)

## UX-10 · P2 · “Back to library” should go to the Library
Reproduced workflow issue · Feed · Effort S (exit fix)

Twice in the sandbox: launch Feed from Settings, advance to the last item, click Back to library → the URL remains ?view=settings. Separately, captured mobile Feed shows a truncated title, 0:00 / 0:00 and Pause while metadata is unavailable. Playback did not initialize in this automation environment; no production playback defect is inferred.

### Suggested change
- Either navigate to Library or rename the action Close feed / Back to Settings according to the actual destination.
- Related state improvement: show explicit Loading video / Cannot play states; do not leave an apparently playable zero-duration transport indefinitely.
- Related layout improvement: keep the video title readable on phones; consider a compact Save control with an accessible label.
- Handle empty/unplayable launcher outcomes explicitly (existing issue 062); verify playback and gestures on real mobile browsers before changing media logic.

### Reproduce / inspect
1. Open Settings, then start Feed from the rail.
2. Advance to the terminal item and click Back to library.
3. Observe that Settings is still the destination.

Acceptance: The exit action's text matches its destination in every launch context; loading, empty and playback errors are distinguishable.
Issue: `meta/issues/make-feed-exit-destination-and-loading-state-explicit.md`
- Screenshot: 36-feed-end-before-return.png — The terminal action promises Library. (Sandbox)
- Screenshot: 37-feed-returned-to-settings.png — Actual destination: Settings. Reproduced twice. (Sandbox)
- Screenshot: 32c-feed-production-mobile.png — Mobile layout observation only; playback itself was not validated. (Production · 390px)

## Duplicate semantics provenance
The dHash contract in UX-02 was established in the preceding deduplication task; see `docs/import-deduplication.md` and ADR-011. Screenshots show labels, not byte/pixel equivalence.

## Legacy retirement
User permits removal; no removal implemented. Preserve shared API/media/auth handling. Clean frontend-build verification is required: the CUDA container relies on prebuilt dist and Atelier smoke can skip when dist is missing. Track in `meta/issues/retire-legacy-ui-and-require-atelier-build.md`.

## Method / limitations
Production 907f8a9 read-only; local real-runtime sandbox 2d9b56d with fake inference and fixture media. Chromium 150, 1440×1000/390×844 plus 320×740 spot-check. Browser-only UI observations; separate source-only legacy inventory. No production writes, deployment or push. No real-mobile/screen-reader certification, full image-query clipboard testing, large-file stress tests, or playback validation. Video metadata remained unavailable in automation despite verified healthy fixture HTTP delivery/FFmpeg decode; not diagnosed as a product playback bug. Recording failed at the CDP layer; before/after screenshots used instead. TDD is not applicable to a prose/screenshot report; validate artifact structure, image data and responsive rendering.
