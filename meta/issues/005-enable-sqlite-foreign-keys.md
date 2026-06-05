# 005 Enable SQLite Foreign Keys

## Priority

P1

## Status

Completed.

- App connections now enable `PRAGMA foreign_keys = ON` during bootstrap.
- The SQLite connection hook also enables foreign keys on each opened connection, including the path that loads `sqlite-vector`.
- Tests cover both bootstrap-time enforcement and the normal `openSQLiteDB(..., "")` path.

## Summary

The schema declares foreign keys, but SQLite does not enforce them unless foreign key support is enabled. Right now the app enables WAL but not foreign key enforcement.

## Why This Matters

- Orphan rows become easier to create silently.
- Future job-pipeline refactors will depend more on relational integrity.
- This is low effort and high value.

## Current Behavior

- `internal/app/bootstrap.go`
- `Bootstrap()` enables WAL and runs migrations.
- It does not enable `PRAGMA foreign_keys = ON`.
- `cmd/imgsearch/main.go`
- The SQLite DSN does not include `_foreign_keys=1`.

## Desired Outcome

- Foreign key enforcement is always enabled for app connections.
- Tests clearly run with the same behavior.

## Suggested Approach

- Enable foreign keys during bootstrap or in the DSN.
- Add at least one regression test that would fail if FKs were off.

## Acceptance Criteria

- App connections run with foreign key enforcement enabled.
- A test proves that an invalid FK write is rejected or the expected cascade behavior occurs.
- No existing tests rely on FK enforcement being off.

All acceptance criteria are satisfied by the current implementation.

## Related Files

- `internal/app/bootstrap.go`
- `cmd/imgsearch/main.go`
- `internal/db/migrations.go`
