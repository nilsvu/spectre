#!/usr/bin/env bash
# Outbox entry 003 — eccentricity-reduction board restructure (stage 3).
#
# Approved in session 2026-08-13:
#  - Merge #7411 into #7413 (one code path); retitle #7413 to carry both
#    halves; the "smooth continuation" item of #7416 also moves there.
#    #7411's board card is archived.
#  - New issue: ecc-control loop has no abort conditions (split from #7416).
#  - New issue: shape map from measured ID horizons (split from #7417).
#  - Split #7418: it keeps the documentation task (retitle); new issue for
#    the accuracy-validation programme.
# New issues are added to project board 20 (Feature parity) with
# Status = Backlog.
#
# Post AFTER entry 002 so the cross-referenced evidence comments exist.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo=sxs-collaboration/spectre

# Project 20 node IDs, resolved 2026-08-13 via the GraphQL read API.
project_id=PVT_kwDOAZoyI84Bej-5
status_field_id=PVTSSF_lADOAZoyI84Bej-5zhY9Ot8
backlog_option_id=f75ad846
item_7411=PVTI_lADOAZoyI84Bej-5zg0MuUA

# Create an issue, add it to board 20 in the Backlog column.
new_backlog_issue() {
  local title=$1 body=$2 url item
  url=$(gh issue create --repo "$repo" --title "$title" --body-file "$body")
  echo "$url"
  item=$(gh project item-add 20 --owner sxs-collaboration --url "$url" \
    --format json --jq .id)
  gh project item-edit --id "$item" --project-id "$project_id" \
    --field-id "$status_field_id" --single-select-option-id "$backlog_option_id"
}

# -- PBJ merge -------------------------------------------------------------
gh issue edit 7413 --repo "$repo" \
  --title "PBJ branching: multiple Levs, and correct state across the branch"
gh issue close 7411 --repo "$repo" --reason "not planned" \
  --comment "$(cat "$here/comment-close-7411.md")"
gh project item-archive 20 --owner sxs-collaboration --id "$item_7411"
echo "https://github.com/$repo/issues/7413 (retitled, absorbs #7411; #7411 card archived)"

# -- New issue: abort conditions (split from #7416) ------------------------
new_backlog_issue \
  "Eccentricity control loop has no abort conditions" \
  "$here/new-issue-abort-conditions.md"

# -- New issue: shape map init (split from #7417) --------------------------
new_backlog_issue \
  "Initialize the shape map from the measured ID horizon coefficients" \
  "$here/new-issue-shape-map-init.md"

# -- Split #7418: keep docs task, new validation issue ---------------------
gh issue edit 7418 --repo "$repo" \
  --title "Document intentional gauge/constraint-damping differences from SpEC"
echo "https://github.com/$repo/issues/7418 (retitled)"
new_backlog_issue \
  "Validate accuracy vs SpEC where schemes differ structurally (time stepping, filtering, AMR)" \
  "$here/new-issue-validation.md"
