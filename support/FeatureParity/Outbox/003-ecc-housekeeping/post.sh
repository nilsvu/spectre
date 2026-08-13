#!/usr/bin/env bash
# Outbox entry 003 — eccentricity-reduction housekeeping from the dedupe
# pass (approved in session 2026-08-13):
#  - close #6460 (duplicate of #7416's Lev-schedule half)
#  - close #5892 (the CLI shipped: `spectre bbh eccentricity-control`)
#  - close #7414 (empty body, superseded by the board cluster)
#  - comment on #5938 linking it to #7415 and PR #6009
#  - comment on #5937 narrowing it to nonzero-target ecc + BNS/BHNS
#  - comment on PR #6224 proposing the re-scope to SimulationSupport
#
# Post AFTER entries 001 and 002 (comments reference the evidence threads
# and the new abort-conditions issue).
#
# No project-board operations here: verified via the GraphQL read API
# (2026-08-13) that none of #6460, #5892, #7414, #5937, #5938 is an item
# on project board 20.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo=sxs-collaboration/spectre

gh issue close 6460 --repo "$repo" --reason "not planned" \
  --comment "$(cat "$here/comment-close-6460.md")"
gh issue close 5892 --repo "$repo" --reason "completed" \
  --comment "$(cat "$here/comment-close-5892.md")"
gh issue close 7414 --repo "$repo" --reason "not planned" \
  --comment "$(cat "$here/comment-close-7414.md")"
gh issue comment 5938 --repo "$repo" --body-file "$here/comment-5938.md"
gh issue comment 5937 --repo "$repo" --body-file "$here/comment-5937.md"
gh pr comment 6224 --repo "$repo" --body-file "$here/comment-rescope-6224.md"
