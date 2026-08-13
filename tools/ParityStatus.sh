#!/usr/bin/env bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Feature-parity campaign status, computed from disk and GitHub — never
# recorded anywhere. Exists so no prose document has to assert live state.
# Read-only; safe to run anytime; tolerates a broken network or token.
#
# Usage: tools/ParityStatus.sh

set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUTBOX="$REPO/support/FeatureParity/Outbox"
SCRATCH_DIR=${SCRATCH:-/capstor/scratch/cscs/nilsvu}
UPSTREAM=sxs-collaboration/spectre
PROJECT_NUMBER=20
export GH_PAGER=cat

echo "== repo =="
git -C "$REPO" log --oneline -1
dirty=$(git -C "$REPO" status --porcelain | wc -l)
echo "branch $(git -C "$REPO" branch --show-current), dirty files: $dirty"

echo
echo "== outbox (awaiting the user) =="
pending=0
for entry in "$OUTBOX"/[0-9][0-9][0-9]-*/; do
  [[ -d "$entry" ]] || continue
  pending=$((pending + 1))
  name=$(basename "$entry")
  age=$(( ($(date +%s) - $(stat -c %Y "$entry")) / 86400 ))
  printf '  %s  (%sd old' "$name" "$age"
  [[ -f "$entry/post.sh" ]] || printf ', MISSING post.sh'
  [[ -f "$entry/body.md" ]] || printf ', MISSING body.md'
  printf ')\n'
done
[[ $pending -eq 0 ]] && echo "  empty"

echo
echo "== github tracker ($UPSTREAM) =="
if ! timeout 20 gh api user --jq .login > /dev/null 2>&1; then
  echo "  UNREACHABLE: gh auth or network failed — tracker state unknown"
else
  board=$(timeout 30 gh api graphql -f query='
    query($org: String!, $number: Int!) {
      organization(login: $org) { projectV2(number: $number) {
        items(first: 100) { totalCount nodes {
          content {
            ... on Issue { number title state }
            ... on PullRequest { number title state }
            ... on DraftIssue { title }
          }
          fieldValues(first: 10) { nodes {
            ... on ProjectV2ItemFieldSingleSelectValue {
              name field { ... on ProjectV2SingleSelectField { name } } }
          } }
        } }
      } }
    }' -f org="${UPSTREAM%%/*}" -F number="$PROJECT_NUMBER" 2> /dev/null)
  if [[ -z "$board" ]] || ! grep -q '"projectV2":{' <<< "$board"; then
    echo "  project $PROJECT_NUMBER: NOT VISIBLE — token lacks org-level"
    echo "  'Projects: read' (resource owner must be ${UPSTREAM%%/*})"
  else
    printf '%s' "$board" | python3 -c '
import json, sys
p = json.load(sys.stdin)["data"]["organization"]["projectV2"]
if p is None:
    print("  project: NOT VISIBLE — token lacks org Projects: read")
    sys.exit()
total = p["items"]["totalCount"]
nodes = p["items"]["nodes"]
print("  project board: %d items" % total)
if total > len(nodes):
    print("  (showing first %d only)" % len(nodes))
by_status = {}
for it in nodes:
    c = it.get("content") or {}
    status = "(no status)"
    fields = {}
    for fv in it["fieldValues"]["nodes"]:
        if not fv:
            continue
        fname = fv.get("field", {}).get("name")
        if fname == "Status":
            status = fv["name"]
        elif fname in ("Priority", "Size"):
            fields[fname] = fv["name"]
    num = c.get("number")
    ref = "#%s" % num if num else "draft"
    title = (c.get("title") or "?")[:60]
    extra = ""
    if fields:
        extra = " (" + ", ".join(
            "%s:%s" % (k[0], fields[k])
            for k in ("Priority", "Size") if k in fields) + ")"
    by_status.setdefault(status, []).append(ref + " " + title + extra)
order = ["Discuss", "In review", "In progress", "Ready", "Backlog", "Done"]
for status in order + sorted(set(by_status) - set(order)):
    rows = by_status.get(status)
    if not rows:
        if status == "Ready":
            print("  [Ready] EMPTY — nothing groomed for implementation")
        elif status == "Discuss":
            print("  [Discuss] EMPTY — nothing awaiting team settlement")
        continue
    if status in ("Backlog", "Done") and len(rows) > 5:
        print("  [%s] %d items (last 5 in board order)"
              % (status, len(rows)))
        rows = rows[-5:]
    else:
        print("  [%s]" % status)
    for row in rows:
        print("    " + row)
' 2> /dev/null || echo "  project parse failed"
  fi
  echo
  echo "  -- open PRs by nilsvu --"
  pr_tmpl='{{range .}}  #{{.number}} {{.title}}'
  pr_tmpl+='{{if .isDraft}} [draft]{{end}}'
  pr_tmpl+=' (checks: {{range .statusCheckRollup}}'
  pr_tmpl+='{{if eq .conclusion "FAILURE"}}FAIL {{end}}{{end}})'
  pr_tmpl+='{{"\n"}}{{end}}'
  timeout 30 gh pr list -R "$UPSTREAM" --author nilsvu --state open \
    --json number,title,isDraft,statusCheckRollup \
    --template "$pr_tmpl" 2> /dev/null || echo "  (query failed)"
  rate=$(timeout 10 gh api rate_limit --jq .resources.core.remaining \
    2> /dev/null || echo '?')
  echo "  rate limit remaining: $rate"
fi

echo
echo "== local lanes (fp-* worktrees on scratch) =="
found_lane=0
for wt in "$SCRATCH_DIR"/spectre-worktrees/fp-*/; do
  [[ -d "$wt" ]] || continue
  found_lane=1
  branch=$(git -C "$wt" branch --show-current 2> /dev/null || echo '?')
  wt_dirty=$(git -C "$wt" status --porcelain 2> /dev/null | wc -l)
  ahead=$(git -C "$wt" rev-list --count develop..HEAD 2> /dev/null \
    || echo '?')
  echo "  $(basename "$wt"): $branch, +$ahead commits vs develop," \
    "dirty: $wt_dirty"
done
[[ $found_lane -eq 0 ]] && echo "  none"

echo
echo "== builds ($SCRATCH_DIR/spectre-builds) =="
found_build=0
for bd in "$SCRATCH_DIR"/spectre-builds/feature-parity/build-*/ \
  "$SCRATCH_DIR"/spectre-builds/fp-*/build-*/; do
  [[ -d "$bd" ]] || continue
  found_build=1
  mtime=$(stat -c %y "$bd" 2> /dev/null | cut -d. -f1)
  echo "  ${bd#"$SCRATCH_DIR"/spectre-builds/}  ($mtime)"
done
[[ $found_build -eq 0 ]] && echo "  none"
