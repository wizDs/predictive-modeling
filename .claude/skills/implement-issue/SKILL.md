---
name: implement-issue
description: Implement a linked GitHub issue end-to-end - branch from main, implement, validate locally then against CI, get an independent code review, simplify, and open the PR. Use when the user links a GitHub issue (URL or #number) and asks to implement/fix/ship it, or invokes /implement-issue <issue>.
---

# Implement Issue

Drives one GitHub issue from link to open PR through a fixed pipeline, with two bounded
retry loops (validation, review) so a stuck issue surfaces back to the user instead of
spinning forever.

Pipeline: **read issue -> branch -> implement -> validate (loop) -> review (loop) ->
/simplify -> gh pr create**

## 0. Setup

- Resolve the issue: `gh issue view <url-or-number> --json title,body,number,labels,url`.
  If no issue is given, ask for one — this skill needs a concrete target, not a vague task.
- `git status` first. If there are uncommitted changes that aren't yours to discard, stop and
  ask the user (stash, commit, or abort) before touching branches.
- `git fetch origin && git checkout -b <branch> origin/main`. Name the branch from the issue:
  `fix/<number>-<slug>` for bug-labeled issues, `feat/<number>-<slug>` otherwise — match
  whatever prefix convention `git log --oneline -20` shows for this repo.

## 1. Implement

Read the issue's Problem/Evidence/Proposed-fix sections (or equivalent) as the spec. Do the
implementation work yourself — don't delegate understanding of the issue to a subagent.
Commit as you go with real messages; don't wait until the end for one giant commit.

**Scope the blast radius from the knowledge graph first** (best-effort — skip silently if
`graphify-out/graph.json` doesn't exist, or if the repo is small enough to just hold in
context). Instead of grepping blind for what the change touches:

- `graphify query "<the thing the issue changes>"` — returns the affected nodes and their
  `source_location`s, so you read only the files that actually matter.
- `graphify path "A" "B"` — traces the dependency path between two concepts (e.g. does this
  CI change reach the Pants resolve?).

Lean on the graph's *cross-community* and *surprising* edges — they catch couplings grep
misses, so you implement every spot the first time instead of discovering a missed file in
review. This is the same navigation the issue-author should use when scoping a change up
front; here it keeps the implementation complete.

## 2. Validate — local first, push last

Run this repo's own CI checks locally before ever pushing, so iteration is fast:

- `mypy (src)`: `pants check src::` (or `pants --changed-since=origin/main
  --changed-dependents=transitive --tag='-app' check` for just the touched slice).
- `mypy (apps/<app>)`, for each touched app: `cd apps/<app> && uv sync --group dev && uv run
  mypy --explicit-package-bases .`
- `.github/scripts` test suite (if touched): `cd .github/scripts && uv sync --group dev && uv
  run pytest -q`.

If a command fails: read the failure, fix it, re-run **only the failing command(s)**, repeat.
Cap at **5 local-fix attempts**; if still red after 5, stop and summarize the blocker for the
user rather than continuing to guess.

Once local checks are green, push and let the real CI have the final word — this repo's
`ci.yml` only triggers on `push: [main]` or `pull_request: [main]`, so a feature-branch push
alone won't run it. Open the PR now (`gh pr create --draft` if the pipeline's later steps
aren't done yet) specifically to get that `pull_request` trigger, then:

- `gh pr checks --watch` (or `gh run watch` on the run it kicks off).
- If CI is red but local was green (environment drift, missing secret, etc.), fix and push
  again, re-watch. Same 5-attempt cap, counted separately from the local-fix budget.
- Don't open a second PR later for the "final" `gh pr create` in step 5 — mark this one ready
  (`gh pr ready`) instead, updating title/body if they were placeholders.

## 3. Code review — independent subagent

Once CI is green, get a review that hasn't seen your implementation reasoning:

```
Agent({
  subagent_type: "general-purpose",
  description: "Independent review of <branch>",
  prompt: "Review the diff between origin/main and <branch> in <repo path> (`git diff
    origin/main...<branch>`) for correctness bugs, not style — this implements issue #<n>:
    <issue url>. You have no context beyond the diff and the issue; read both cold. Report
    concrete, verified findings only (file:line, failure scenario), or say it's clean."
})
```

If it reports real findings: fix them (back to step 2's validate loop — re-run the relevant
local checks before re-pushing), then review again. Cap at **3 review rounds**; if still
finding real issues after 3, stop and hand the findings to the user instead of iterating
further solo.

If it comes back clean, proceed.

## 4. Simplify

Run `/simplify` on the branch's diff. If it changes anything, that's a code change — loop
back to step 2 (re-validate locally, re-push, confirm CI still green) before moving on.

## 5. Finalize

- If a draft PR was already opened in step 2, fill in a real title/body and `gh pr ready` it.
- Otherwise, `gh pr create` now with a title/body summarizing the issue and the fix, and a
  `Closes #<n>` line so merging auto-closes the issue.
- **Refresh the knowledge graph** so it reflects the change you just shipped (best-effort —
  skip if graphify isn't installed): `/graphify . --update` re-extracts only the changed
  files. The versioned graph artifacts live in `graphify-out/` (`graph.json`, `graph.html`,
  `GRAPH_REPORT.md`, `manifest.json`; the `cache/` and machine-local sidecars are gitignored).
  If any of those tracked artifacts changed, commit them (`git commit -m "chore: refresh
  graphify graph"`) and push to the PR branch so the map stays in sync with the code.
- Report the PR URL back to the user. Don't merge it — that's the user's call.
