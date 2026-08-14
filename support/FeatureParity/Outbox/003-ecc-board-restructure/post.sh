#!/usr/bin/env bash
# Outbox entry 003 — eccentricity-reduction board restructure (stage 3).
#
# Approved in session 2026-08-13:
#  - Merge #7411 into #7413 (one code path); retitle #7413 to carry
#    both halves; archive #7411's board card.
#  - New issues (abort conditions, shape-map init, validation split
#    from #7418) — born in the Discuss column with Priority/Size set:
#    they carry design proposals with open points.
#  - Split #7418: it keeps the documentation task (retitle).
#  - Define Priority options High/Medium/Low (skipped if the field
#    already has options).
#  - Surveyed cards move Backlog -> Discuss with Priority/Size set
#    (their survey comments are posted by entry 002; #7412 is already
#    in Discuss and gets Priority/Size only).
#
# Post AFTER entry 002 so the cross-referenced evidence comments exist.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo=sxs-collaboration/spectre

# Project 20 node IDs, resolved 2026-08-13/14 via the GraphQL read API.
project_id=PVT_kwDOAZoyI84Bej-5
status_field_id=PVTSSF_lADOAZoyI84Bej-5zhY9Ot8
discuss_option_id=5a061dc8
priority_field_id=PVTSSF_lADOAZoyI84Bej-5zhY9PBY
size_field_id=PVTSSF_lADOAZoyI84Bej-5zhY9PBc
size_xs=6c6483d2 size_s=f784b110 size_m=7515a9f1 size_l=817d0097
item_7411=PVTI_lADOAZoyI84Bej-5zg0MuUA
item_7412=PVTI_lADOAZoyI84Bej-5zg0MqXg
item_7413=PVTI_lADOAZoyI84Bej-5zg0MqtU
item_7415=PVTI_lADOAZoyI84Bej-5zg0MrAk
item_7416=PVTI_lADOAZoyI84Bej-5zg0MuyM
item_7417=PVTI_lADOAZoyI84Bej-5zg0Mu0c
item_7418=PVTI_lADOAZoyI84Bej-5zg0MyMU

edit() { # item-id field-id option-id
  gh project item-edit --id "$1" --project-id "$project_id" \
    --field-id "$2" --single-select-option-id "$3" > /dev/null
}

# -- Priority options: define once, look up IDs ---------------------------
n_opts=$(gh api graphql -f query='query($id: ID!) { node(id: $id) {
  ... on ProjectV2SingleSelectField { options { id } } } }' \
  -f id="$priority_field_id" --jq '.data.node.options | length')
if [[ "$n_opts" -eq 0 ]]; then
  gh api graphql -f query='mutation($fieldId: ID!) {
    updateProjectV2Field(input: {fieldId: $fieldId, singleSelectOptions: [
      {name: "High",   color: RED,    description: ""},
      {name: "Medium", color: YELLOW, description: ""},
      {name: "Low",    color: GRAY,   description: ""}
    ]}) { projectV2Field { ... on ProjectV2SingleSelectField { id } } } }' \
    -f fieldId="$priority_field_id" > /dev/null
  echo "Priority options created: High / Medium / Low"
fi
prio_json=$(gh api graphql -f query='query($id: ID!) { node(id: $id) {
  ... on ProjectV2SingleSelectField { options { id name } } } }' \
  -f id="$priority_field_id" --jq '.data.node.options')
prio_high=$(jq -r '.[] | select(.name == "High").id' <<< "$prio_json")
prio_medium=$(jq -r '.[] | select(.name == "Medium").id' <<< "$prio_json")
prio_low=$(jq -r '.[] | select(.name == "Low").id' <<< "$prio_json")

# -- PBJ merge -------------------------------------------------------------
# (#7413 retitle and #7411 close were done by the user directly,
# 2026-08-14; only the card archive remains.)
gh project item-archive 20 --owner sxs-collaboration --id "$item_7411"
echo "#7411 card archived"

# -- Split #7418: keep docs task -------------------------------------------
gh issue edit 7418 --repo "$repo" \
  --title "Document intentional gauge/constraint-damping differences from SpEC"
echo "https://github.com/$repo/issues/7418 (retitled)"

# -- New issues: born in Discuss with Priority/Size ------------------------
new_discuss_issue() { # title body-file priority-option size-option
  local url item
  url=$(gh issue create --repo "$repo" --title "$1" --body-file "$here/$2")
  echo "$url"
  item=$(gh project item-add 20 --owner sxs-collaboration --url "$url" \
    --format json --jq .id)
  edit "$item" "$status_field_id" "$discuss_option_id"
  edit "$item" "$priority_field_id" "$3"
  edit "$item" "$size_field_id" "$4"
}
new_discuss_issue \
  "Eccentricity control loop has no abort conditions" \
  new-issue-abort-conditions.md "$prio_high" "$size_xs"
new_discuss_issue \
  "Initialize the shape map from the measured ID horizon coefficients" \
  new-issue-shape-map-init.md "$prio_medium" "$size_s"
new_discuss_issue \
  "Validate accuracy vs SpEC where schemes differ structurally (time stepping, filtering, AMR)" \
  new-issue-validation.md "$prio_medium" "$size_l"

# -- Surveyed cards: -> Discuss, Priority, Size ----------------------------
# #7412 is already in Discuss (entry 001): Priority/Size only.
edit "$item_7412" "$priority_field_id" "$prio_high";   edit "$item_7412" "$size_field_id" "$size_s"
edit "$item_7413" "$status_field_id" "$discuss_option_id"
edit "$item_7413" "$priority_field_id" "$prio_high";   edit "$item_7413" "$size_field_id" "$size_l"
edit "$item_7415" "$status_field_id" "$discuss_option_id"
edit "$item_7415" "$priority_field_id" "$prio_medium"; edit "$item_7415" "$size_field_id" "$size_m"
edit "$item_7416" "$status_field_id" "$discuss_option_id"
edit "$item_7416" "$priority_field_id" "$prio_medium"; edit "$item_7416" "$size_field_id" "$size_s"
edit "$item_7417" "$status_field_id" "$discuss_option_id"
edit "$item_7417" "$priority_field_id" "$prio_low";    edit "$item_7417" "$size_field_id" "$size_m"
edit "$item_7418" "$status_field_id" "$discuss_option_id"
edit "$item_7418" "$priority_field_id" "$prio_low";    edit "$item_7418" "$size_field_id" "$size_s"
echo "surveyed cards in Discuss with Priority/Size set"
