# 047: Harden Scripts/Tooling Portability and Wire Script Tests

## Priority

P2

## Status

Open.

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

- [ ] Add a portable file-size helper for Darwin and Linux in `import_images.sh`.
- [ ] Resolve ONNX Runtime library paths based on both OS and architecture.
- [ ] Remove path-splitting hazards from release packaging scripts.
- [ ] Add a `mise` task for script tests.
- [ ] Decide whether `mise run test` should include script tests or add a documented full-test task.

## Related Files

- `scripts/import_images.sh`
- `scripts/import_images_4chan_test.sh`
- `scripts/resolve_onnxruntime_lib.sh`
- `scripts/package_release.sh`
- `scripts/run_imgsearch_cuda_container_test.sh`
- `mise.toml`
