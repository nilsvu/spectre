#!/usr/bin/env bash
# Outbox entry 001 — the initial-orbital-parameters thread (stage 1).
#
# Reviewed by the user 2026-08-13. Self-contained: posts before all
# other entries and depends on none of them.
#  - Survey + design comment on #7412.
#  - The three actions that comment proposes:
#    - re-scope comment on PR #6224 (PN kernels into SimulationSupport)
#    - narrowing comment on #5937 (nonzero-target ecc + BNS/BHNS)
#    - close #5892 (the CLI shipped: `spectre bbh eccentricity-control`)
# No project-board operations: #7412 stays in Backlog; none of the
# other issues has a board card (verified via the read API 2026-08-13).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo=sxs-collaboration/spectre

gh issue comment 7412 --repo "$repo" --body-file "$here/comment-7412.md"
gh pr comment 6224 --repo "$repo" --body-file "$here/comment-rescope-6224.md"
gh issue comment 5937 --repo "$repo" --body-file "$here/comment-5937.md"
gh issue close 5892 --repo "$repo" --reason "completed" \
  --comment "$(cat "$here/comment-close-5892.md")"
