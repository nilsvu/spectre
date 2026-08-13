#!/usr/bin/env bash
# Outbox entry 001 — eccentricity-reduction survey: evidence comments.
#
# Posts one survey-evidence comment on each surveyed board issue. The
# comment on #7413 covers the merged PBJ scope (it absorbs #7411 — the
# retitle and close are entry 002). Post 001 before 002 so the
# cross-references in 002 land on threads that already carry evidence.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo=sxs-collaboration/spectre

gh issue comment 7412 --repo "$repo" --body-file "$here/comment-7412.md"
gh issue comment 7413 --repo "$repo" --body-file "$here/comment-7413.md"
gh issue comment 7415 --repo "$repo" --body-file "$here/comment-7415.md"
gh issue comment 7416 --repo "$repo" --body-file "$here/comment-7416.md"
gh issue comment 7417 --repo "$repo" --body-file "$here/comment-7417.md"
gh issue comment 7418 --repo "$repo" --body-file "$here/comment-7418.md"
