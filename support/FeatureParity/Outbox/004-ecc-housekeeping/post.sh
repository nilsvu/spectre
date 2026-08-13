#!/usr/bin/env bash
# Outbox entry 004 — eccentricity-reduction housekeeping from the
# dedupe pass (stage 4):
#  - close #6460 (duplicate of #7416's Lev-schedule half)
#  - close #7414 (empty body, superseded by the board cluster)
#  - comment on #5938 linking it to #7415 and PR #6009
#
# (The #6224 re-scope, #5937 narrowing, and #5892 close moved to
# entry 001 as dependents of the #7412 thread.)
#
# Post AFTER entries 002 and 003: the comments reference the evidence
# threads and the new abort-conditions issue.
#
# No project-board operations: verified via the GraphQL read API
# (2026-08-13) that neither #6460, #7414, nor #5938 is an item on
# project board 20.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo=sxs-collaboration/spectre

gh issue close 6460 --repo "$repo" --reason "not planned" \
  --comment "$(cat "$here/comment-close-6460.md")"
gh issue close 7414 --repo "$repo" --reason "not planned" \
  --comment "$(cat "$here/comment-close-7414.md")"
gh issue comment 5938 --repo "$repo" --body-file "$here/comment-5938.md"
