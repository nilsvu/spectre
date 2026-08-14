<!-- Distributed under the MIT License. -->
<!-- See LICENSE.txt for details. -->

# Outbox — staged GitHub writes

Every GitHub write an agent wants to happen is staged here as one
directory and posted by the user from their own shell (outside the
container, where write credentials live). Agents never post.

```
Outbox/
  NNN-<slug>/
    post.sh    # the exact commands, runnable from any checkout root
    body.md    # the issue/PR/comment body post.sh references
    ...        # extra bodies or attachments if the entry needs them
```

- `NNN` is sequential, three digits, never reused (check `git log` for
  the high-water mark when the directory is empty).
- `post.sh` is complete and reviewable: `set -euo pipefail`, resolves
  the repo root from its own location, prints the URL of what it
  created. One entry = one logical write (an issue, a PR, a batch of
  related comments).
- A PR entry's `post.sh` pushes the lane branch to the `nilsvu/spectre`
  fork and opens a **draft** PR against `sxs-collaboration/spectre`
  `develop`. Lane worktrees share refs with the main checkout, so the
  push works from the user's own shell without entering scratch:
  `git -C /users/nilsvu/spectre push origin fp/<slug>`. A PR entry is
  staged only after the co-review with the user is done; `body.md`'s
  testing section records what was run.
- A PR `body.md` links its issue with a closing keyword
  (`Closes #NNNN`, first line after the summary). This drives the
  board: the card's "Linked pull requests" field shows implementation
  state, and the merge auto-moves the card to Done (project workflow).
  One PR per issue; if a PR deliberately covers only part of an issue,
  say so and use `Part of #NNNN` instead — no closing keyword.
- Entries must be self-contained: a teammate (or `sxs-bot`, later)
  could post them without any agent context. Entries embedding
  resolved GitHub node IDs (project boards) assume prompt posting.
- Housekeeping closes of other people's issues may be staged when the
  evidence is uncontroversial — the user has the authority and reviews
  every entry before posting.
- Staging commits may bypass the repo's source-file hooks
  (`git commit --no-verify`): entry payloads are GitHub bodies
  (markdown tables, quoted code), not repo source.
- Every comment and issue body ends with the attribution footer

  ```
  ---

  🤖 drafted by: [Claude Fable 5](https://claude.com/claude-code)
  (and reviewed by me 🙋)
  ```

  so agent-drafted text is attributed correctly when the user posts it
  from their own account. (Name the model that drafted the entry; the
  second line is the user's review statement — the user reviews every
  body before posting.)

## Writing comment and issue bodies

Read this section before drafting any body — worker, orchestrator, or
reviewer. These guidelines were converged with the user (2026-08-13,
eccentricity-reduction survey) and define what a good entry looks like.

**Purpose.** One comment per issue carries survey evidence + design
proposal + open points (manual, pipeline stages 1–2). It must be enough
for a team meeting to discuss, and phrased so a single follow-up
comment settling the open points makes the issue ready for
implementation — ideally with little iteration left for the PR.

**Structure**, in order:

1. Title line (`# Survey: …`).
2. Header: surveyed revisions (SpEC/SpECTRE/other repos @ commit), so
   `file:line` references stay meaningful.
3. Verdict/headline first — what is at parity, what is not.
4. Findings: mechanism comparison with `file:line` in both codebases;
   record negative results ("searched X, found nothing").
5. Prior art: actionable rows only (read-first, revive, close-as-dup),
   minor merged PRs collapsed to one line. Always write issue/PR
   numbers as `#NNNN` — including in table cells — so GitHub renders
   them as links.
6. `## Proposed design` at implementation depth: name the files,
   knobs, and mechanisms; say what is pipeline-side vs executable-side;
   include a **Testing / acceptance** paragraph.
7. `## Open points to settle` — a task list with the number written
   into the label (`- [ ] **1. Name** — options…`; GitHub swallows
   ordinal markers on task-list items, so `1. [ ]` renders without
   numbers), each with the options and a recommendation where the
   evidence supports one. Close with the standard line: *"A follow-up
   comment settling these points makes this issue ready for
   implementation (→ Ready)."*
8. The attribution footer (above).

**Length and readability** (the compromise that survived review):

- As long as the issue warrants, no longer. Both failure modes were
  rejected in review: exhaustive walk-throughs that bury the design,
  and prose so dense it cannot be scanned.
- One idea per bullet, bold lead-ins; no multi-fact paragraph prose.
- **No hard line wrapping**: GitHub renders single newlines in issue
  and PR comments as line breaks, so wrapped source displays ragged.
  Write each paragraph and list item as one long line (code blocks and
  tables excepted). The repo's 80-column style does not apply to
  bodies — they are GitHub content, not source.
- No verbatim quote blocks or mechanism walk-throughs — a `file:line`
  reference replaces them. Exception: short code anchors (≤ ~6 lines,
  a few per comment) where the exact text carries the point.
- Bodies must be self-contained for the team: cite only repo-relative
  `file:line` at the pinned revisions, in repositories the team can
  access (spectre, SpEC, SimulationSupport). **Never reference local
  paths** (scratch, home, container) — teammates cannot see them. The
  full-detail survey report on scratch is session-internal working
  material; nothing on GitHub points at it.

**Tone:**

- State real defects plainly; never soften them.
- Do not inflate incidental friction into blockers — label decisions
  as decisions ("a few decisions to settle", not "the real blocker")
  so reviewers are not scared off routine items.
- Distinguish verified facts from inference; flag uncertain claims,
  and hedge honestly where the implementer must verify a detail.

## Posting flow (user)

```sh
# from your own shell, in any checkout of the feature-parity branch:
bash support/FeatureParity/Outbox/NNN-<slug>/post.sh
```

Review the entry first; edit freely — the files are the proposal, not
a done deal. After posting, tell an agent (or run `/parity-outbox`):
it verifies the write landed via the read API, then removes the entry
directory in a commit whose message records the created URL. GitHub
holds the truth from that moment; the outbox drains to empty.

Mechanics:

- `post.sh` stops at the first failure (`set -euo pipefail`). Before
  re-running a partially posted entry, comment out the lines that
  already succeeded — re-running reposts them as duplicates.
  `/parity-outbox` (or the read API) tells you what landed.
- Project-board commands (`gh project item-add/item-edit/item-archive`)
  need the `project` scope on your token; grant once with
  `gh auth refresh -s project`.
