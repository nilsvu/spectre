#!/usr/bin/env bash
# Outbox entry 010 — bug issue: BOOTSTRAP_PY_DEPS deps invisible on
# Debian-patched pips. Found during #7447; deliberately split out of
# that PR (round-6 relaxation). Plain repo issue, no board card.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo=sxs-collaboration/spectre

gh issue create --repo "$repo" \
  --title "BOOTSTRAP_PY_DEPS installs packages where nothing looks for them on Debian-patched pips" \
  --body-file "$here/body.md"
