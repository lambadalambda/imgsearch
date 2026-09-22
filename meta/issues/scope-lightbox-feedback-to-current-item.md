# Scope Lightbox action feedback to the current item

## Summary

UX06 review found retained media-action feedback across item changes: Lightbox uses one createMediaActions instance, and its error/pending state is not reset when a different pin opens. A failed action on one item can leave that item's error visible on another.

## Acceptance Criteria

- Error/success/pending feedback is clearly associated with the operation and item that produced it.
- Add deferred/failed-action regressions for next/previous, close/reopen and responses arriving after switching items.
- Do not change edit-discard policy or rewrite shared action handling unnecessarily.

## Notes

- Existing limitation, not a demonstrated UX06 layout regression. Track separately from the new Save changes/More action hierarchy.
