# search

## Git workflow

- Start each piece of work on a feature branch off an up-to-date `main` (e.g. `feat/<issue>-<slug>`).
- Commit each meaningful, working slice of implementation as you go, rather than one large commit at the end.
- After a PR merges: `git checkout main && git pull` so local `main` matches the remote.

## Agent skills

### Issue tracker

Issues live in GitHub Issues on `dygwon/search` (via `gh`). See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
