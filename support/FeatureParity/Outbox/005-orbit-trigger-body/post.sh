#!/usr/bin/env bash
# Outbox entry 005 — update #7415's issue body: fold in the PN-time
# alternative from #5938 (nilsdeppe's suggestion, adopted).
#
# Design change vs the posted body: termination after ~N orbits happens
# via a PN T(N) estimate + the existing TimeCompares machinery (Part A,
# pipeline only, resolves #5938); the stateless NTimesPerOrbit trigger
# remains for observation cadence (Part B); the EveryNOrbits trigger is
# dropped. Open point 1 asks the team to confirm the split.
#
# The user may instead paste the body via the web UI (same file).
# Suggested issue retitle to match the narrowed trigger scope (optional,
# not scripted): "Orbit-based observation cadence and PN-time run length"
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo=sxs-collaboration/spectre

gh issue edit 7415 --repo "$repo" --body-file "$here/body-7415.md"
echo "https://github.com/$repo/issues/7415 (body updated)"
