#!/usr/bin/env bash
# Outbox entry 006 — #7447 "Bin directories": survey + design proposal.
#
# Approved in session 2026-08-18:
#  - Replace the thin #7447 body with the survey + design proposal
#    (7 numbered open points); card moves to Discuss, Size M.
#    (Design simplified 2026-08-18: the CLI self-locates, so only the
#    first schedule() needs plumbing — Size dropped from L.)
#  - Close #5951 as consolidated into #7447 (same problem, 2024; its
#    design discussion is folded into the proposal).
#  - Close #7444 as duplicate of #7443 (title is verbatim a bullet of
#    #7443's body; empty body). Neither #5951 nor #7444 is a board card.
#
# Priority for #7447 is High — set the native issue field in the web UI
# after posting (no scripted path for issue fields).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo=sxs-collaboration/spectre

# Project 20 node IDs, resolved 2026-08-18 via the GraphQL read API.
project_id=PVT_kwDOAZoyI84Bej-5
status_field_id=PVTSSF_lADOAZoyI84Bej-5zhY9Ot8
discuss_option_id=5a061dc8
size_field_id=PVTSSF_lADOAZoyI84Bej-5zhY9PBc
size_m_option_id=7515a9f1
item_7447=PVTI_lADOAZoyI84Bej-5zg0NA8M

# -- #7447: survey + design proposal becomes the issue body ----------------
gh issue edit 7447 --repo "$repo" --body-file "$here/body.md"
echo "https://github.com/$repo/issues/7447 (body replaced)"

# Card: Status -> Discuss, Size -> M
gh project item-edit --id "$item_7447" --project-id "$project_id" \
  --field-id "$status_field_id" --single-select-option-id "$discuss_option_id"
gh project item-edit --id "$item_7447" --project-id "$project_id" \
  --field-id "$size_field_id" --single-select-option-id "$size_m_option_id"
echo "#7447 card: Discuss, Size M"

# -- #5951: consolidated into #7447 ----------------------------------------
gh issue close 5951 --repo "$repo" --reason "not planned" \
  --comment "$(cat "$here/comment-close-5951.md")"
echo "https://github.com/$repo/issues/5951 (closed, consolidated into #7447)"

# -- #7444: duplicate of #7443 ---------------------------------------------
gh issue close 7444 --repo "$repo" --reason "not planned" \
  --comment "$(cat "$here/comment-close-7444.md")"
echo "https://github.com/$repo/issues/7444 (closed, duplicate of #7443)"

echo "REMINDER: set Priority: High on #7447 (native issue field, web UI)."
