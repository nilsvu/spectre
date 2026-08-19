#!/usr/bin/env bash
# Outbox entry 009 — draft PR for #7447 "Bin directories" + the
# settlement addendum comment.
#
# Lane: fp/bin-directories, three commits on develop (06fa7dffb1),
# HEAD 5d90c1a454:
#   1b6e3d73b6 Let the CLI wrapper find the Python package next to
#              itself
#   2486fee1e6 Run scheduled simulations from a bin directory
#   5d90c1a454 Copy later pipeline steps' executables to the bin
#              directory
# Co-review done 2026-08-18 over six rounds. Final scope per the
# revised settlement (comment below): simulation-local bin directory
# with ancestor sharing across Levs; deps-freezing, Env.sh, Manifest,
# and the pip bootstrap fix all relaxed out (the pip bug is outbox
# 010); guard as build-shape check. 19 files, +1450/-178; short commit messages, no session trailer. Affected
# tests 5/5 + build-dir-gone check exit 0; Bbh pipeline tests must
# run in CI (noted in the PR body).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo=sxs-collaboration/spectre

# Push the lane branch (worktree shares refs with the main checkout).
git -C /users/nilsvu/spectre push origin fp/bin-directories

gh pr create --repo "$repo" --base develop \
  --head nilsvu:fp/bin-directories --draft \
  --title "Run scheduled simulations from a bin directory" \
  --body-file "$here/body.md"

# Settlement addendum on the issue (3 narrowed, 5 deferred to #7443,
# 6 as build-shape check).
gh issue comment 7447 --repo "$repo" \
  --body-file "$here/comment-7447-settlement.md"

# Card: Status -> In review (project 20 node IDs, resolved 2026-08-18).
gh project item-edit --id PVTI_lADOAZoyI84Bej-5zg0NA8M \
  --project-id PVT_kwDOAZoyI84Bej-5 \
  --field-id PVTSSF_lADOAZoyI84Bej-5zhY9Ot8 \
  --single-select-option-id df73e18b
echo "#7447 card: In review"
