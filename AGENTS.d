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
- Keep dependencies minimal and justify each new dependency.
- Document architectural decisions and trade-offs in `docs/`.
- Keep `CHANGELOG.md` up to date for user-visible changes and notable internal behavior changes.
- Use `mise` to manage developer tool versions in `mise.toml`.
- Use `mise run <task>` as the standard task runner for common workflows.
- Run formatting and tests before every commit.

## Collaboration
- Preserve user data and avoid destructive operations by default.
- Make incremental changes that are easy to review.
- When unsure, choose the simplest approach that can ship an MVP.
