#!/usr/bin/env bash
# Outbox entry 009 — draft PR for #7447 "Bin directories".
#
# Lane: fp/bin-directories, single commit 2c136a767b on develop
# (06fa7dffb1). Co-review done 2026-08-18 over three fix rounds
# (reviewer findings + all user review directives: no copy_executable
# compat, copy-executables naming, create_bin kept through Next on
# measured evidence, ancestor-shared bin across Levs bounded by
# DirectoryStructure formats, verbatim-copyable wrapper with single
# PYTHONPATH composition + equivalence test, Env.sh naming). Tests
# 5/5 affected suite; Bbh pipeline tests must run in CI (noted in
# the PR body).
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
