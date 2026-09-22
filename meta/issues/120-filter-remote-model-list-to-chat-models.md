# Filter the remote model list to chat-capable models

## Summary

"Fetch models" on the Atelier Settings page against a Lemonade server (`http://192.168.1.198:13305/v1`) appeared to return only image-generation models. The server does return the chat and vision models too, but `/v1/models` lists image models first alphabetically, the page auto-fills the model field with the first id, and the browser datalist then only suggests ids that start with that text.

## Requirements

- When the `/models` entries carry `labels` (Lemonade), drop models that are only for image generation, upscaling, transcription, or embedding, and list vision-capable chat models first.
- Keep servers without labels working unchanged.
- Make the page explain that clearing the field shows every fetched model.

## Acceptance Criteria

- Listing against a labelled server yields chat/vision models only, vision first; unlabelled servers are untouched (unit test).
- The auto-filled model is a vision-capable one when the server labels models.

## Notes

- Reported 2026-09-22.
