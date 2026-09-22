# Announce confirmation consequences when dialogs open

## Summary

Shared ConfirmDialog has an accessible name and focus trap but no aria-describedby association for its body. Consequences and retention/irreversibility text are readable after navigation but may not be announced when the dialog opens.

## Acceptance Criteria

- Associate concise consequence text with confirmation dialogs without duplicating or over-announcing long arbitrary content.
- Add an accessibility-tree regression for destructive confirmation counts/policy and retain focus-trap/Escape behavior.

## Notes

- Nonblocking shared-component follow-up from UX01/02 review (`frontend/src/components/ConfirmDialog.svelte`). Not a data-safety defect in the transactional cleanup endpoint.
