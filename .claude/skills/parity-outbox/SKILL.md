---
name: parity-outbox
description: >-
  Manage the feature-parity outbox — list and validate staged GitHub
  writes, hand the user exact posting commands, verify posted entries
  via the read API, and clean them up. Never posts anything itself.
allowed-tools: ["Bash", "Read", "Grep", "Glob", "Edit", "Write"]
---

The outbox convention is `support/FeatureParity/Outbox/README.md`.
This skill has three jobs; do the ones the current request needs:

1. **List and validate**: for each `Outbox/NNN-<slug>/` entry, one
   line — what it posts, age, and any defects: missing or non-runnable
   `post.sh` (`bash -n`), missing `body.md`, a PR entry whose lane
   branch is missing/dirty/behind `develop`, a stale reference to an
   issue or PR that has changed on GitHub since staging (check via
   read-only `gh`). Report defects; fix only formatting-level ones
   yourself.
2. **Hand over**: print the exact command per entry, ready to paste
   into the user's own shell (outside the container):

   ```sh
   bash support/FeatureParity/Outbox/NNN-<slug>/post.sh
   ```

   Never run `post.sh` and never work around the posting boundary — a
   permission denial here is the security model working.
3. **Verify and clean**: when the user says an entry was posted (or a
   status check suggests it), confirm via the read API that the
   issue/PR/comment exists and matches, then remove the entry
   directory in a local commit whose message records the created URL.
   Never push. If verification fails, say so and leave the entry in
   place.
