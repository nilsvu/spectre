#!/usr/bin/env bash
# Outbox entry 001 — the initial-orbital-parameters thread (stage 1),
# REMAINDER. Already posted by the user via the web UI (2026-08-13):
#  - survey + design comment on #7412
#    https://github.com/sxs-collaboration/spectre/issues/7412#issuecomment-5287320896
#  - close #5892 with comment
#    https://github.com/sxs-collaboration/spectre/issues/5892#issuecomment-5287373871
# Remaining writes, self-contained, no board operations:
#  - re-scope comment on PR #6224 (PN kernels into SimulationSupport)
#  - narrowing comment on #5937 (nonzero-target ecc + BNS/BHNS)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo=sxs-collaboration/spectre

gh pr comment 6224 --repo "$repo" --body-file "$here/comment-rescope-6224.md"
gh issue comment 5937 --repo "$repo" --body-file "$here/comment-5937.md"
