#!/usr/bin/env bash
# Outbox entry 002 — eccentricity-reduction survey: evidence comments
# (stage 2).
#
# Posts one survey-evidence comment on each remaining surveyed board
# issue (the #7412 thread went out with entry 001). The comment on
# #7413 covers the merged PBJ scope (it absorbs #7411 — the retitle
# and close are entry 003). Post AFTER 001 (the #7416 comment cites
# the #7412 survey comment) and BEFORE 003 (whose cross-references
# land on these threads).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo=sxs-collaboration/spectre

gh issue comment 7413 --repo "$repo" --body-file "$here/comment-7413.md"
gh issue comment 7415 --repo "$repo" --body-file "$here/comment-7415.md"
gh issue comment 7416 --repo "$repo" --body-file "$here/comment-7416.md"
gh issue comment 7417 --repo "$repo" --body-file "$here/comment-7417.md"
gh issue comment 7418 --repo "$repo" --body-file "$here/comment-7418.md"
