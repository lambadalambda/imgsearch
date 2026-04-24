# 047: Harden Scripts/Tooling Portability and Wire Script Tests

## Priority

P2

## Status

Completed.

## Summary

Several helper scripts assume specific host tools or architectures, and existing script tests are not part of the default test task.

## Context

- `scripts/import_images.sh:313` uses BSD `stat -f %z`, which fails on GNU/Linux.
- `scripts/resolve_onnxruntime_lib.sh:7` hard-codes the macOS ARM64 ONNX Runtime dylib path.
- `scripts/package_release.sh:193` uses unquoted command substitution from `find`, which can split paths with spaces.
- Existing shell tests are not run by `mise run test`, which currently runs only `go test ./...`.

## Risks

- Bulk import and release packaging can fail on Linux or non-ARM macOS hosts.
- Script regressions are missed during normal verification.
- Release packaging can break on paths with spaces.

## Acceptance Criteria

- [x] Add a portable file-size helper for Darwin and Linux in `import_images.sh`.
- [x] Resolve ONNX Runtime library paths based on both OS and architecture.
- [x] Remove path-splitting hazards from release packaging scripts.
- [x] Add a `mise` task for script tests.
- [x] Decide whether `mise run test` should include script tests or add a documented full-test task.

## Resolution

- Replaced BSD-only `stat -f %z` video sizing with a portable `wc -c` helper.
- Made ONNX Runtime library resolution explicit by OS/architecture and fail clearly when the pinned dependency does not ship a matching test library.
- Replaced unquoted Linux release `find` command substitution with array accumulation to preserve paths containing spaces.
- Added `scripts/resolve_onnxruntime_lib_test.sh`, `mise run test:scripts`, and `mise run test:full`.
- Kept `mise run test` as the fast Go-only suite and documented script/full test commands in `docs/development.md`.

## Related Files

- `scripts/import_images.sh`
- `scripts/import_images_4chan_test.sh`
- `scripts/resolve_onnxruntime_lib.sh`
- `scripts/package_release.sh`
- `scripts/run_imgsearch_cuda_container_test.sh`
- `mise.toml`
