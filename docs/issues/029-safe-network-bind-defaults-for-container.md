# 029: Default Container Bind Address to Loopback

## Priority

P1

## Status

Open.

## Summary

The container startup script defaults to binding the HTTP server on `0.0.0.0:8080`, which exposes the service by default.

## Context

- `scripts/run_imgsearch_cuda_container.sh` currently sets:
  - `-addr "${IMGSEARCH_ADDR:-0.0.0.0:8080}"`

## Risks

- Unintended network exposure in local or shared environments.
- Magnifies impact of missing API authentication.

## Acceptance Criteria

- [ ] Default container bind address is loopback-only.
- [ ] Public binding requires explicit user opt-in.
- [ ] Deployment docs call out auth/reverse-proxy requirements when exposed.

## Related Files

- `scripts/run_imgsearch_cuda_container.sh`
- `README.md`
