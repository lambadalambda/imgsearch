# Add loading, empty-state, and accessibility polish

## Summary

The interface has a strong foundation but needs clearer first-load, loading, empty, contrast, and reduced-motion polish.

## Requirements

- Add explicit loading or skeleton states for gallery, videos, tags, and results.
- Add more actionable empty states with direct calls to upload, clear filters, or retry where appropriate.
- Recheck muted text contrast, especially paths, captions, and tiny metadata.
- Add a dedicated focus color token instead of reusing processing-state color.
- Ensure reduced-motion removes transforms as well as transitions.
- Replace ASCII `...` with a polished ellipsis or icon control.

## Acceptance Criteria

- Loading and empty states are visually distinct from loaded content.
- Empty states suggest the next useful action.
- Focus outlines and processing states are visually distinct.
- Reduced-motion users do not receive hover/CTA translation effects.
- Small muted text meets practical contrast targets.

## Notes

- This issue intentionally groups polish items that can be tackled incrementally.
