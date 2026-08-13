---
name: parity-status
description: >-
  Observe the feature-parity campaign — computed live state, GitHub
  tracker, outbox, local lanes. The session-boundary entry point;
  also the orchestrator's observe step.
allowed-tools: ["Bash", "Read", "Grep", "Glob"]
---

Run `tools/ParityStatus.sh` and report on it. Do not re-derive what it
prints; add judgment on top:

1. **Awaiting the user**: outbox entries, phrased as the posts the
   user owes, with the exact command for each; plus any prepared
   `sbatch` handovers.
2. **On GitHub**: campaign PRs with new review comments or red CI;
   gap/design issues with new discussion; board moves. Use read-only
   `gh` queries for anything the script summarizes too coarsely.
3. **In flight locally**: lane worktrees and what state their last
   commit/dirty files suggest; anything to harvest from finished
   workers or detached runs (check sentinel files before concluding).
4. **Anomalies** — only if present: a lane branch behind `develop` by
   weeks, a stale outbox entry (> 7 days), rate limit nearly
   exhausted, the project board unreadable (token permission), dirty
   harness worktree.

Verify anything surprising against disk before reporting it (read the
log, not the filename). End with the single most useful next action.
