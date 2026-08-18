#!/usr/bin/env bash
# Outbox entry 007 — comment on #7416: per-Lev bin-directory
# multiplication, discovered during the #7447 implementation.
# Approved in session 2026-08-18 (discovery triage).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo=sxs-collaboration/spectre

gh issue comment 7416 --repo "$repo" --body-file "$here/comment-7416.md"
