#!/usr/bin/env bash
# Outbox entry 009 — draft PR for #7447 "Bin directories".
#
# Lane: fp/bin-directories, three commits on develop (06fa7dffb1),
# HEAD b6a44be71e:
#   f202210171 Find the Python dependencies where pip put them,
#              from one wrapper
#   7ca0fee90d Run scheduled simulations from a bin directory
#   b6a44be71e Copy later pipeline steps' executables to the bin
#              directory
# Co-review done 2026-08-18 over four fix rounds (reviewer findings +
# all user review directives; simulation mirrors the build layout,
# bin/ + lib/; ancestor-shared bin across Levs; Env.sh naming; fixes
# the pre-existing BOOTSTRAP_PY_DEPS invisibility on Debian pips).
# Tests 5/5 affected suite; Bbh pipeline tests must run in CI (noted
# in the PR body).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo=sxs-collaboration/spectre

# Push the lane branch (worktree shares refs with the main checkout).
git -C /users/nilsvu/spectre push origin fp/bin-directories

gh pr create --repo "$repo" --base develop \
  --head nilsvu:fp/bin-directories --draft \
  --title "Run scheduled simulations from a bin directory" \
  --body-file "$here/body.md"

# Card: Status -> In review (project 20 node IDs, resolved 2026-08-18).
gh project item-edit --id PVTI_lADOAZoyI84Bej-5zg0NA8M \
  --project-id PVT_kwDOAZoyI84Bej-5 \
  --field-id PVTSSF_lADOAZoyI84Bej-5zhY9Ot8 \
  --single-select-option-id df73e18b
echo "#7447 card: In review"
