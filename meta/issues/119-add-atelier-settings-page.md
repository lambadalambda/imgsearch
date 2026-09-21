# Add the Atelier Settings page

## Summary

The Rail shows a disabled "Settings (soon)" button (`frontend/src/components/Rail.svelte:147`). Add a `settings` view, following the Statistics page pattern, that edits annotation backend settings.

## Requirements

- `ViewMode` gains `"settings"`, URL-synced as `?view=settings`, with a Rail button and header breadcrumb.
- Form sections: Backend (radio: Native / Remote server). Native: variant selector `e4b` / `26b`, disabled with an explanation when custom native paths are pinned by flags. Remote: base URL, API key (password field, shows "key is set" state, blank keeps existing), model name (text with datalist from a "Fetch models" call to `/v1/models` when available), timeout, concurrency.
- "Test connection" button calling `POST /api/settings/annotation/test`, showing success or the returned error inline.
- Save with inline validation errors from the API; unsaved-changes indicator.
- "Re-annotate all" button using the existing `ConfirmDialog`, with a note that it queues every image and video.
- Status line showing the currently active backend and model from the API, refreshed after save.
- Keyboard and focus handling consistent with the Statistics page; touch targets at least 44px on mobile.

## Acceptance Criteria

- Playwright smoke: open Settings from the Rail, switch to Remote, fill fields, Test connection against the stub, Save, reload, verify persisted values (key masked), trigger Re-annotate all and confirm the dialog.
- `npm run check` passes.

## Notes

- Subissue of [113](113-annotation-backend-settings-page.md). Depends on 115 and 118 for the API; can be built against a stub before 117 lands.
