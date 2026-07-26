---
name: address-pr-comments
description: Read a reviewer's inline comments on an arbitrary GitHub PR, act on each one - implement it, file a tracking issue if it's a TODO for later, or answer it if it's a question - reply on the thread, then commit and push to the PR branch. Use when the user links a PR and asks to address/handle/implement review comments, or invokes /address-pr-comments <pr>.
---

# Address PR Comments

Turns a reviewer's inline comments into either a code change, a filed issue, or an answered
question — with a reply posted on every thread so re-running this skill later only picks up
what's actually new.

## 0. Setup

- PR reference is required (`<number>` or full URL). If missing, ask for one.
- Resolve owner/repo from the PR URL, or use the current repo's remote if only a number was
  given. `gh pr checkout <n>` (add `-R owner/repo` for a PR outside the current repo's remote —
  if that repo isn't cloned anywhere locally, say so and stop rather than guessing a path).
- `git status` first, same as always — don't check out over uncommitted work that isn't yours
  to discard.

## 1. Fetch and thread the comments

```
gh api repos/{owner}/{repo}/pulls/<n>/comments --jq '.[] | {id, in_reply_to_id, path, line, body, user: .user.login}'
```

Group by thread: a comment with `in_reply_to_id: null` starts a thread; every comment whose
`in_reply_to_id` points to it (directly or transitively) is a reply. **Skip any thread that
already has a reply** — that's this skill's own resumability marker, so a second run against
the same PR only touches genuinely new comments. Also skip threads not authored by a reviewer
whose feedback the user wants acted on (if the PR has multiple reviewers, ask which ones count,
unless it's obvious — e.g. only one person has ever commented).

## 2. Classify each unreplied thread's root comment

- **Question** (asks something, doesn't instruct a change) — goes to step 3a.
- **Actionable now** (a concrete, small, in-scope instruction — "use X instead of Y", "make
  this more concise", "don't do this") — goes to step 3b.
- **Open-ended / future work** (explicitly asks for a ticket, or describes a bigger
  investigation than fits in this PR — "make an issue for this", "look into whether...") —
  goes to step 3c.
- **Ambiguous or needs a product/architecture call only the user can make** — don't guess.
  Collect it and surface it in the final summary instead of replying.

## 3a. Questions — answer, don't implement

Investigate the actual code/repo state before answering — a plausible-sounding guess posted
to a real PR thread is worse than no reply. Reply with a direct, complete answer:

```
gh api repos/{owner}/{repo}/pulls/<n>/comments/<comment_id>/replies -f body="<answer>"
```

If answering it surfaces a genuinely open question beyond this PR's scope, it's fine to both
answer *and* file a follow-up issue (step 3c) for the open part — say so in the reply.

## 3b. Actionable comments — implement, then reply

Make the change, run whatever local validation is cheap and relevant (don't re-run a full CI
matrix per comment — batch that after step 4), then reply once committed:

```
git add <files> && git commit -m "..."
gh api repos/{owner}/{repo}/pulls/<n>/comments/<comment_id>/replies -f body="Done in <short-sha>."
```

## 3c. TODO / future-investigation comments — file an issue

```
gh issue create --title "..." --body "..."
gh api repos/{owner}/{repo}/pulls/<n>/comments/<comment_id>/replies -f body="Filed #<issue-number> to track this."
```
Write the issue like the repo's existing ones (check `gh issue list` for house style —
Problem/Evidence/Proposed-fix sections, links to related issues/PRs) rather than a bare
one-liner.

## 4. Validate, commit, push

Once every actionable thread has a code change, run this repo's actual CI checks locally
(same as `implement-issue`'s validate step: `pants check`, per-app `uv run mypy`, `.github`'s
`uv run pytest`, or whatever this repo/PR's checks are) before pushing. Fix anything broken,
re-check, then:

```
git push
```

Comment replies from steps 3a/3b/3c can go out as you go rather than being batched to the end
— there's no ordering requirement between "reply on GitHub" and "push the branch."

## 5. Summarize

Report back per comment: implemented / issue filed (with link) / answered (with link) /
flagged for the user (with why it wasn't safe to guess). Don't resolve review threads
yourself — that's the reviewer's call once they've seen the replies.
