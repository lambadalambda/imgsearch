# Add a persisted settings store and `/api/settings`

## Summary

Configuration is flags and a few env vars only. A settings page needs a persisted store shared by the API and worker processes, plus an endpoint to read and update it.

## Requirements

- Migration adding a `settings` table (`key TEXT PRIMARY KEY, value_json TEXT, updated_at TEXT`) and a monotonically increasing `settings_version` (a row or a `PRAGMA user_version` style counter) that writers bump so other processes can detect changes cheaply.
- `internal/settings` package with a typed `AnnotationSettings` struct: `Backend` (`native` | `openai`), `NativeVariant` (`e4b` | `26b`), `OpenAI{BaseURL, APIKey, Model, TimeoutSeconds, Concurrency}`; `Load`, `Save`, and `Version` functions; validation (URL parse, non-empty model when backend is `openai`, variant whitelist).
- On first run, seed the annotation settings from the startup flags so existing deployments behave identically.
- `GET /api/settings` returns the settings with `api_key` replaced by a boolean `api_key_set`; `PUT /api/settings` accepts a full object, treats an absent or empty `api_key` as "keep existing", validates, saves, bumps the version, and returns the masked result.
- `POST /api/settings/annotation/test` performs a small remote request (models list or a tiny chat completion) against the supplied settings without saving and returns success or the error text.

## Acceptance Criteria

- Handler tests cover GET masking, PUT validation errors, PUT keep-existing-key semantics, and version bump.
- Settings survive restart and are visible from a second `-mode=worker` process.

## Notes

- Subissue of [113](113-annotation-backend-settings-page.md).
- Store the key in plaintext in the local SQLite DB; document that in the README security section.
