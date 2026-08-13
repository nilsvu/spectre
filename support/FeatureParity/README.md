<!-- Distributed under the MIT License. -->
<!-- See LICENSE.txt for details. -->

# The SpEC feature-parity campaign — operating manual

This is the STATIC operating manual: mission, pipeline, state model,
security model, procedures. It contains **no live state by design** (a
rule inherited from the GPU campaign, which learned it by being burned:
every hand-maintained "current state" block went stale within days).

**Live state is computed:** `tools/ParityStatus.sh` (repo, outbox,
GitHub tracker, local lanes — read-only, safe anytime).
**The tracker is GitHub:** issues and PRs in `sxs-collaboration/spectre`
and the org project
[Feature parity #20](https://github.com/orgs/sxs-collaboration/projects/20).
**Staged GitHub writes:** `Outbox/` (see `Outbox/README.md`).

# Mission

Bring SpECTRE binary-black-hole simulations to production quality and
feature parity with SpEC, the battle-proven predecessor code (full
checkout: `/users/nilsvu/spec`). The actual gap list is established by
surveying both codes — umbrella issue
[#6563](https://github.com/sxs-collaboration/spectre/issues/6563) is
guidance for where to look (eccentricity reduction, ringdown
transition, cut-X, AMR, RWZ extraction, …) but is dated and
incomplete; gaps become goals only through surveys and user approval.

This campaign is a team effort with strong human-in-the-loop control.
Agents gather evidence, implement approved designs, and pre-review;
the user decides, designs, publishes, and — with the team — reviews
every line of code in PRs.

# The pipeline

Five stages. Every `⇒` is a human act (the user posting an outbox
entry or acting on GitHub); agents work autonomously only *inside* a
stage the user opened, never across a gate.

The project board's columns ARE the pipeline state, and the user moves
the cards: **Backlog** (captured, thin) → **Ready** (design approved,
groomed for implementation) → **In progress** → **In review** →
**Done**.

1. **Survey + design proposal** — the assignment is co-written in
   session, then the worker runs it autonomously: SpEC vs SpECTRE on
   its area, mechanism by mechanism, `file:line` evidence in both
   codebases, existing issues/PRs checked first (repo-wide, not just
   the board; housekeeping proposals — close/link/consolidate — are a
   standard deliverable). The board already carries a decomposition of
   #6563 into granular issues — drafts deepen those issues, new issue
   bodies only where none exists. Each per-issue draft carries the
   evidence AND a concrete design proposal with **numbered open
   points**, each listing the options and a recommendation
   where the evidence supports one — so a single comment is enough to
   discuss at a team meeting. Structure, length, and tone are defined
   in `Outbox/README.md` ("Writing comment and issue bodies") — read
   before drafting. Results land in the session; the user decides
   which gaps advance ⇒ one comment per issue: survey + proposal +
   open points.
2. **Settle** — team discussion (meetings, issue threads) settles the
   substance of the numbered open points; the user records the
   settlement in a follow-up GitHub comment and has the final say. A
   settling comment may explicitly defer a named point ("deferred, not
   blocking") — deferred points do not block Ready. Hard forks can
   still be co-designed in session first (the issue's worker answers
   deep-dive questions as input), but the record of the decision is
   the settling comment. When every open point is settled or
   explicitly deferred, the user grooms the issue to the Ready column.
   The team learns this convention from the user directly — no process
   note is posted to GitHub.
3. **Implement** — once its issue is in *Ready*, the worker executes
   it in its own lane worktree: code + tests per the repo rules,
   affected tests plus `ctest -L unit` before declaring done.
   Deliverable: a lane branch with clean history + a PR body draft
   (motivation, user-visible effects, testing performed). Autonomous,
   because the design it implements is already the user's.
4. **Co-review** — the reviewer agent produces findings; user and
   session walk the diff together; the issue's worker applies fixes
   until the user is satisfied ⇒ PR posted (draft, against upstream
   `develop`).
5. **Team review** — the user or a teammate reviews every line on
   GitHub, approves, merges. The worker only prepares responses to
   review comments, staged back through the outbox.

Validation runs attach wherever needed: survey evidence, PR testing
sections, post-merge regression checks.

The process itself is expected to change — edit this manual when it
does; no changelog is kept.

# State model (three layers)

- **Rarely changing** — this repo, branch `feature-parity` (the user
  pushes it): this manual, `.claude/agents/`,
  `.claude/skills/parity-*`, `tools/ParityStatus.sh`, `Outbox/`
  staging.
- **Daily changing** — GitHub: all campaign state — gap and design
  issues, the project-20 board, PRs, review threads.
- **Current** — computed on demand, recorded nowhere:
  `tools/ParityStatus.sh`.

Campaign state — gap lists, designs, task status, discussion — lives
on GitHub, never in this repo. The repo branch carries only the
harness.

# Security model (non-negotiable)

- **Agents have no GitHub write path.** The container's `gh` token is
  read-only and ssh to GitHub does not work from the container. Keep
  it that way: never store write credentials or ssh keys where agents
  run.
- **Every GitHub write is staged in `Outbox/` and posted by the user**
  from their own shell (outside the container), where their ssh key
  and write-capable `gh` auth live.
- **`.claude/settings.json` denies `git push` and `gh` write
  commands.** These are tripwires against accidents, not the security
  boundary — the boundary is the absence of credentials. A denied
  command is never worked around.
- **Slurm submissions are user actions.** Agents prepare packages and
  hand over the exact `sbatch` command.
- Agents commit harness-branch changes (outbox staging and cleanup)
  locally; they never push.
- Planned relaxations, in order, each on explicit user GO:
  1. a fine-grained PAT scoped to the `nilsvu/spectre` fork only
     (`contents: write`) so agents can push lane branches; PRs and
     comments stay user-posted;
  2. the `sxs-bot` machine account (org agreement pending) posts
     outbox entries under its own name — the outbox format already
     anticipates a machine consumer — and agent PR review expands
     only then.

# Decision boundary

The user decides: campaign goals and scope, which surveyed gaps
advance, every design, every GitHub publication, implementation
acceptance in co-review, and (with the team) the merge. Stage
transitions are never agent acts.

Agents decide only execution details inside a dispatched assignment: how to
search, what evidence to collect, how to structure code that
implements an approved design, test mechanics. In session, the
assistant proposes — agenda, assignments, staging — and acts on approval.

Mechanical user actions (posting outbox entries, pushing, `sbatch`)
are handed over as exact commands, not questions.

# Workers

One worker per issue — or per area for surveys, where an area is a
named cluster of board issues (e.g. eccentricity reduction). The
issue is the worker's identity and its memory:

- **Within a session, the same worker follows its issue through the
  pipeline stages** — continue it with follow-up assignments rather
  than spawning fresh; its accumulated SpEC/SpECTRE context is an
  asset.
- **Across sessions, agents do not survive; the issue thread
  rehydrates a fresh worker** (evidence comments, design, review
  discussion). This is one more reason deliverables are posted, not
  held in agent context.
- Every assignment states its deliverable and a stop condition
  (including when to stop *unsuccessfully*) before it starts; the
  stop condition is not renegotiated mid-flight.
- Discoveries outside the worker's issue are reported for in-session
  triage — never folded in, never chased.
- Validation and measurement runs carry preregistered success
  criteria in the assignment, adjudicated against logs/artifacts on
  disk; compute-node work is delivered as a frozen package plus the
  exact submit command for the user.

# Implementation lanes

- Branch `fp/<slug>` off `upstream/develop` (never off this harness
  branch — PRs must stand alone against upstream):

  ```sh
  git worktree add /capstor/scratch/cscs/nilsvu/spectre-worktrees/fp-<slug> \
    -b fp/<slug> develop
  ln -s /users/nilsvu/spectre/CMakeUserPresets.json \
        /capstor/scratch/cscs/nilsvu/spectre-worktrees/fp-<slug>/
  # configure with -D USE_GIT_HOOKS=OFF; build dirs land on scratch
  # via the preset's ${sourceDirName}-derived binaryDir
  ```

- Follow the repo rules (`.claude/rules/`), build only needed targets,
  run affected tests (`ctest -L unit` for regressions).
- Co-review precedes every PR staging. Human PR review on GitHub is
  the real gate — agent review is preparation for it, never a
  substitute.

# Runs and validation

In scope from the start: unit tests, input-file tests, pipeline tests
(`support/Pipelines/`), regression tests, short simulations, and long
runs.

- **Login node** (shared, be courteous): the container kills any
  process at 14400 CPU-core-seconds (`RLIMIT_CPU`, SIGKILL) — segment
  long work. No Slurm tools in the container; never conclude "no
  jobs" from a failed `squeue`.
- **Compute nodes**: user-submitted Slurm packages. Compute nodes here
  are GH200 (GPU) nodes; whether long validation runs use the GPU
  campaign's code, this machine's CPUs, or a CPU cluster is an open
  scope question — ask when it first matters, don't assume.
- The GPU campaign's manual
  (`/users/nilsvu/spectre.worktrees/gpu/support/GpuPort/README.md`)
  holds measured container facts (stack limits, OpenBLAS `+p < 128`,
  EDF submission, scratch layout) and the hermetic-package
  discipline — consult it before preparing compute-node work.
- Every run gets its own directory under
  `/capstor/scratch/cscs/nilsvu/Runs/` (per `AGENTS.local.md`).
  Detach anything longer than a tool-call timeout and leave a
  sentinel file; verify results against logs on disk, not against
  reports.

# Where knowledge lives

| file / dir | role |
|---|---|
| `support/FeatureParity/README.md` | this manual — static, no live state |
| `support/FeatureParity/Outbox/` | staged GitHub writes awaiting the user |
| `tools/ParityStatus.sh` | computed live state |
| GitHub issues + project 20 | the tracker: gaps, designs, tasks |
| GitHub PRs | implementations in flight |
| `.claude/agents/{orchestrator,worker,reviewer}.md` | the roles |
| `.claude/skills/parity-{status,outbox}/` | observation; posting flow |

Any assistant memory store is a cache of this repo and GitHub, never a
source of truth.

# How to continue from here

Start a session as the orchestrator — it knows this procedure:

```sh
cd /users/nilsvu/spectre.worktrees/feature-parity
claude --agent=orchestrator
```

or observe only: `tools/ParityStatus.sh` (any shell) or
`/parity-status` (any session). The orchestrator observes, summarizes
what moved on GitHub, proposes an agenda, and executes what the user
approves: co-writing assignments, dispatching workers, co-design, co-review,
staging outbox entries, handing over exact commands.
