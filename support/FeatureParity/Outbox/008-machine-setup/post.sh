#!/usr/bin/env bash
# Outbox entry 008 — #7443 "Simplify setup on a known machine":
# survey + design proposal (9 open points). Approved in session
# 2026-08-18. Card moves to Discuss, Size L.
#
# Priority for #7443 is Medium — set the native issue field in the
# web UI after posting (no scripted path for issue fields).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo=sxs-collaboration/spectre

# Project 20 node IDs, resolved 2026-08-18 via the GraphQL read API.
project_id=PVT_kwDOAZoyI84Bej-5
status_field_id=PVTSSF_lADOAZoyI84Bej-5zhY9Ot8
discuss_option_id=5a061dc8
size_field_id=PVTSSF_lADOAZoyI84Bej-5zhY9PBc
size_l_option_id=817d0097
item_7443=PVTI_lADOAZoyI84Bej-5zg0M-64

gh issue edit 7443 --repo "$repo" --body-file "$here/body.md"
echo "https://github.com/$repo/issues/7443 (body replaced)"

gh project item-edit --id "$item_7443" --project-id "$project_id" \
  --field-id "$status_field_id" --single-select-option-id "$discuss_option_id"
gh project item-edit --id "$item_7443" --project-id "$project_id" \
  --field-id "$size_field_id" --single-select-option-id "$size_l_option_id"
echo "#7443 card: Discuss, Size L"

echo "REMINDER: set Priority: Medium on #7443 (native issue field, web UI)."
