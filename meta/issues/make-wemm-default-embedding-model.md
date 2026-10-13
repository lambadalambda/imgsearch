# Make WeMM-Embedding-2B Q4_K_M the default embedding model

## Summary

WeMM-Embedding-2B beats Qwen3-VL-Embedding-2B and 8B on retrieval and is faster (see `docs/embedding-model-evaluation.md`). Q4_K_M keeps quality (84.1 vs 84.2 T2I@1). Make it the default for new installs and document the switch for existing ones.

## Requirements

- Default model/mmproj paths, download URLs, dimensions (2048) and `-llama-native-prompt-format wemm` in `default_model_assets.go`, `mise.toml` serve tasks, release `run.sh` and container docs.
- Decide where the GGUF is hosted: no official GGUF exists.
- Keep Qwen3-VL usable via flags; document that switching re-embeds the library.

## Acceptance Criteria

- A fresh start downloads WeMM and passes the fixture integration tests.
- README, CHANGELOG and the evaluation doc describe the new default and the re-embed cost.

## Notes

- [The llama.cpp bump](bump-llama-cpp-october-2026.md) was evaluated and not adopted; ship on the current pin.
- With `wemm`, query/passage instructions are unused but still part of the model version string; drop them from the version for `wemm` before this ships.
