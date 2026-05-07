# Separate selection mode from pagination controls

## Summary

Selection and destructive image actions currently live in the same control cluster as pagination. This increases visual clutter and the risk of destructive misclicks.

## Requirements

- Show a contextual selection toolbar only when selection mode is active.
- Keep pagination focused on previous/page/next navigation.
- Visually separate destructive actions from navigation.
- Keep `Delete all` behind a stronger affordance and confirmation.

## Acceptance Criteria

- Normal browsing shows pagination without selection/delete clutter.
- Selection mode clearly shows selected count, clear/select-page, and delete-selected actions.
- Destructive actions are styled as destructive and still require confirmation.
- Existing bulk delete behavior remains tested.

## Notes

- This is a high-priority safety improvement from the Claude review.
