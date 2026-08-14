#!/usr/bin/env bash
# Outbox entry 003 — eccentricity-reduction board restructure,
# REMAINDER (verified against GitHub 2026-08-14).
#
# Already done by the user directly: #7413 retitled; #7411 closed;
# new issues created and placed in Discuss with Sizes — #7497
# (termination conditions), #7498 (shape-map init), #7499 (validation);
# surveyed cards moved to Discuss with Sizes.
#
# Remaining:
#  - define Priority options High/Medium/Low (guarded: skipped if the
#    field already has options)
#  - set Priority on the nine Discuss cards
#  - retitle #7418 (docs task; the validation half lives in #7499)
#  - archive #7411's board card
set -euo pipefail
repo=sxs-collaboration/spectre

# Project 20 node IDs, resolved 2026-08-13/14 via the GraphQL read API.
project_id=PVT_kwDOAZoyI84Bej-5
priority_field_id=PVTSSF_lADOAZoyI84Bej-5zhY9PBY
item_7411=PVTI_lADOAZoyI84Bej-5zg0MuUA
item_7412=PVTI_lADOAZoyI84Bej-5zg0MqXg
item_7413=PVTI_lADOAZoyI84Bej-5zg0MqtU
item_7415=PVTI_lADOAZoyI84Bej-5zg0MrAk
item_7416=PVTI_lADOAZoyI84Bej-5zg0MuyM
item_7417=PVTI_lADOAZoyI84Bej-5zg0Mu0c
item_7418=PVTI_lADOAZoyI84Bej-5zg0MyMU
item_7497=PVTI_lADOAZoyI84Bej-5zg2ji2c
item_7498=PVTI_lADOAZoyI84Bej-5zg2jj7Y
item_7499=PVTI_lADOAZoyI84Bej-5zg2jkdo

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

set_priority() { # item-id option-id
  gh project item-edit --id "$1" --project-id "$project_id" \
    --field-id "$priority_field_id" --single-select-option-id "$2" > /dev/null
}
set_priority "$item_7412" "$prio_high"
set_priority "$item_7413" "$prio_high"
set_priority "$item_7497" "$prio_high"
set_priority "$item_7415" "$prio_medium"
set_priority "$item_7416" "$prio_medium"
set_priority "$item_7498" "$prio_medium"
set_priority "$item_7499" "$prio_medium"
set_priority "$item_7417" "$prio_low"
set_priority "$item_7418" "$prio_low"
echo "priorities set on the nine Discuss cards"

# -- Retitle #7418 (docs task) ---------------------------------------------
gh issue edit 7418 --repo "$repo" \
  --title "Document intentional gauge/constraint-damping differences from SpEC"
echo "https://github.com/$repo/issues/7418 (retitled)"

# -- Archive #7411's board card --------------------------------------------
gh project item-archive 20 --owner sxs-collaboration --id "$item_7411"
echo "#7411 card archived"
