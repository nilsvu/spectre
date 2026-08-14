#!/usr/bin/env bash
# Outbox entry 004 — REMAINDER (verified against GitHub 2026-08-14).
#
# Done by the user directly: #6460, #7414, #5938 all closed (no
# comments; the supersession trail lives in the posted survey bodies).
#
# Remaining, one write: the comment on #5938 recording that
# @nilsdeppe's PN-time suggestion was adopted into #7415's design
# (Part A) — closes the thread's loop and credits the suggestion.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo=sxs-collaboration/spectre

gh issue comment 5938 --repo "$repo" --body-file "$here/comment-5938.md"
