#!/usr/bin/env bash
# Outbox entry 009 (remainder) — settlement addendum + card move.
#
# Already posted by the user (verified 2026-08-19 via the read API):
# PR sxs-collaboration/spectre#7507 "Freeze bin directories for
# submitted jobs" from fp/bin-directories, with "Closes #7447".
# Remaining writes:
#  - Settlement addendum comment on #7447 (3 narrowed, 5 deferred to
#    #7443, 6 as build-config check).
#  - Card Status -> In review.
#
# Lane UPDATES after CI fixes are pushed directly (not via outbox):
#   git -C /users/nilsvu/spectre push --force-with-lease origin fp/bin-directories
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo=sxs-collaboration/spectre

gh issue comment 7447 --repo "$repo" \
  --body-file "$here/comment-7447-settlement.md"

# Card: Status -> In review (project 20 node IDs, resolved 2026-08-18).
gh project item-edit --id PVTI_lADOAZoyI84Bej-5zg0NA8M \
  --project-id PVT_kwDOAZoyI84Bej-5 \
  --field-id PVTSSF_lADOAZoyI84Bej-5zhY9Ot8 \
  --single-select-option-id df73e18b
echo "#7447 card: In review"
