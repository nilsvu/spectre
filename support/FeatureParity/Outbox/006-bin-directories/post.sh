#!/usr/bin/env bash
# Outbox entry 006 (remainder) — housekeeping closes for #7447.
#
# Already posted by the user 2026-08-18, verified via the read API:
# #7447 body replaced (survey + design), settling comment (1b 2a 3a
# 4-defer 5a 6a 7a), card In progress, Size M, Priority High.
#
# Remaining write (verified 2026-08-19: #7444 already closed by the
# user; #5951 still open):
#  - Close #5951 as consolidated into #7447.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo=sxs-collaboration/spectre

gh issue close 5951 --repo "$repo" --reason "not planned" \
  --comment "$(cat "$here/comment-close-5951.md")"
echo "https://github.com/$repo/issues/5951 (closed, consolidated into #7447)"
