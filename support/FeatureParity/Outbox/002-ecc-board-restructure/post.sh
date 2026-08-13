#!/usr/bin/env bash
# Outbox entry 002 — eccentricity-reduction board restructure.
#
# Approved in session 2026-08-13:
#  - Merge #7411 into #7413 (one code path); retitle #7413 to carry both
#    halves; the "smooth continuation" item of #7416 also moves there.
#  - New issue: ecc-control loop has no abort conditions (split from #7416).
#  - New issue: shape map from measured ID horizons (split from #7417).
#  - Split #7418: it keeps the documentation task (retitle); new issue for
#    the accuracy-validation programme.
# New issues are added to project board 20 (Feature parity); set their
# Status column by hand (they land without one).
#
# Post AFTER entry 001 so the cross-referenced evidence comments exist.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo=sxs-collaboration/spectre

# -- PBJ merge -------------------------------------------------------------
gh issue edit 7413 --repo "$repo" \
  --title "PBJ branching: multiple Levs, and correct state across the branch"
gh issue close 7411 --repo "$repo" --reason "not planned" \
  --comment "$(cat "$here/comment-close-7411.md")"
echo "https://github.com/$repo/issues/7413 (retitled, absorbs #7411)"

# -- New issue: abort conditions (split from #7416) ------------------------
url=$(gh issue create --repo "$repo" \
  --title "Eccentricity control loop has no abort conditions" \
  --body-file "$here/new-issue-abort-conditions.md")
echo "$url"
gh project item-add 20 --owner sxs-collaboration --url "$url"

# -- New issue: shape map init (split from #7417) --------------------------
url=$(gh issue create --repo "$repo" \
  --title "Initialize the shape map from the measured ID horizon coefficients" \
  --body-file "$here/new-issue-shape-map-init.md")
echo "$url"
gh project item-add 20 --owner sxs-collaboration --url "$url"

# -- Split #7418: keep docs task, new validation issue ---------------------
gh issue edit 7418 --repo "$repo" \
  --title "Document intentional gauge/constraint-damping differences from SpEC"
echo "https://github.com/$repo/issues/7418 (retitled)"
url=$(gh issue create --repo "$repo" \
  --title "Validate accuracy vs SpEC where schemes differ structurally (time stepping, filtering, AMR)" \
  --body-file "$here/new-issue-validation.md")
echo "$url"
gh project item-add 20 --owner sxs-collaboration --url "$url"
