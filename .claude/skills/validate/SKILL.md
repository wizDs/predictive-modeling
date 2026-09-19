---
name: validate
description: Run this repo's CI validation checks locally before pushing — ruff lint, mypy (src) via Pants, per-app mypy, and the .github/scripts tests. Mirrors .github/workflows/ci.yml so green here means green in CI. Use when asked to validate/check/lint/typecheck the working changes locally, during iterative work on a feature branch, or when the user invokes /validate. Pass `full` to check everything instead of just the changed slice.
---

# Validate (local CI mirror)

Runs the same gates as `.github/workflows/ci.yml` on your machine, so you catch failures
in seconds instead of through a push → CI → red round-trip. Faithfully mirrors CI's four
checks: **ruff**, **mypy (src)**, **mypy (apps/\*)**, and the **`.github/scripts` tests**.

By default it checks only the **changed slice** relative to `origin/main` (what CI does on a
`pull_request`) — fast enough for a tight iterative loop. Pass `full` (`/validate full`) to
run the complete check set (what CI does on push to `main`).

This skill is **read-only validation** — it never edits code, commits, or pushes. It reports
pass/fail per gate and stops; fixing is the caller's job.

## 0. Setup

- `git fetch origin --quiet` first — the changed-slice checks diff against `origin/main`, so a
  stale ref would mis-scope them. (A shallow clone with no `origin/main` can't do `--changed-since`;
  if that fails, fall back to `full`.)
- Note the mode: `full` if the user passed it, otherwise `changed` (default).
- Tools come from the repo's own toolchain: `pants` (mypy src), per-app `uv` envs (mypy apps),
  `ruff`, and `uv` for `.github/scripts`. If a tool is missing, report that gate as "could not
  run" rather than silently skipping it.

## 1. ruff — full-repo lint

Ruff is a syntax-level lint needing no project deps, so one scan covers `src/*` and every app
(matches CI's single `ruff-action` job):

```bash
ruff check .
```

Ruff is cheap; always run it in full regardless of mode.

## 2. mypy (src) — via Pants

- **changed mode** (default) — mirrors CI's PR path. `--tag='-app'` excludes app targets from the
  check set without narrowing the dependency-graph traversal, so an app depending on a changed
  `src/*` package still keeps that package in scope:
  ```bash
  pants --changed-since=origin/main --changed-dependents=transitive --tag='-app' check
  ```
- **full mode** — mirrors CI's push path:
  ```bash
  pants check src::
  ```

## 3. mypy (apps/\*) — per affected app, in each app's own uv env

Determine which apps to check the same way CI's `changes` job does, then run each app's mypy in
its own uv-managed environment (apps are **not** checked through Pants — only their affected-set
is computed via Pants' dependency inference):

```bash
# affected apps (changed mode): src/* changes pull in dependent apps transitively
apps=$(pants --changed-since=origin/main --changed-dependents=transitive --tag=app list \
        | python3 .github/scripts/pants_addrs_to_apps.py)
# full mode: every app
# apps=$(pants --tag=app list :: | python3 .github/scripts/pants_addrs_to_apps.py)
```

For each app in `$apps` (skip this whole step if the list is empty in changed mode):

```bash
cd apps/<app> && uv sync --group dev && uv run mypy --explicit-package-bases .
```

Run each app's leg independently and don't stop at the first failure — collect all results
(CI's matrix is `fail-fast: false`).

## 4. .github/scripts tests (only if that dir changed, or in full mode)

CI runs these inside its `changes` job. Run them locally when `.github/scripts/**` is in your
diff (`git diff --name-only origin/main | grep -q '^.github/scripts/'`), or always in full mode:

```bash
cd .github/scripts && uv sync --group dev && uv run pytest -q
```

## 5. Report — the ci-gate verdict

Summarize each gate as pass / fail / skipped (not applicable this diff) / could-not-run, then give
the overall verdict the same way CI's `ci-gate` does: **green only if every gate that ran passed
and the rest were legitimately skipped.** For any failure, show the failing command's output
(file:line, the actual mypy/ruff message) so the caller can fix it — don't paraphrase the error away.

If green: say so plainly — this is the same check set CI will run, so a clean local pass means CI
should pass too (barring environment drift like a missing lockfile sync).
