# Agent Rules

## Development Principles
- Follow TDD: write a failing test first, implement the smallest change, then refactor.
- Follow red -> green -> refactor: tests should fail before writing implementation code unless this is absolutely unworkable for the specific change.
- Use Go as the primary language for backend, worker, and server-side logic.
- Use plain web technologies (HTML/CSS/JavaScript) for the frontend unless changed explicitly.
- Keep commits small and topical; one concern per commit.
- Commit early and often to keep progress visible and reversible.

## Quality and Workflow
- Prefer simple, readable code over clever code.
- Add or update tests for every behavior change.
- Treat UI/runtime regressions the same way: add a failing regression test first when practical, then fix the bug and keep the test so it does not reappear.
- Keep dependencies minimal and justify each new dependency.
- Document architectural decisions and trade-offs in `docs/`.
- When reviews or investigations uncover actionable follow-up work, track it in `docs/issues.md` as a checklist and keep one dedicated note per issue under `docs/issues/` with context, risks, and acceptance criteria.
- Keep `CHANGELOG.md` up to date for user-visible changes and notable internal behavior changes.
- Use `mise` to manage developer tool versions in `mise.toml`.
- Use `mise run <task>` as the standard task runner for common workflows.
- Run formatting and all tests before every commit.

## Collaboration
- Preserve user data and avoid destructive operations by default.
- Make incremental changes that are easy to review.
- When unsure, choose the simplest approach that can ship an MVP.
