<!-- Distributed under the MIT License. -->
<!-- See LICENSE.txt for details. -->

# Outbox — staged GitHub writes

Every GitHub write an agent wants to happen is staged here as one
directory and posted by the user from their own shell (outside the
container, where write credentials live). Agents never post.

```
Outbox/
  NNN-<slug>/
    post.sh    # the exact commands, runnable from any checkout root
    body.md    # the issue/PR/comment body post.sh references
    ...        # extra bodies or attachments if the entry needs them
```

- `NNN` is sequential, three digits, never reused (check `git log` for
  the high-water mark when the directory is empty).
- `post.sh` is complete and reviewable: `set -euo pipefail`, resolves
  the repo root from its own location, prints the URL of what it
  created. One entry = one logical write (an issue, a PR, a batch of
  related comments).
- A PR entry's `post.sh` pushes the lane branch to the `nilsvu/spectre`
  fork and opens a **draft** PR against `sxs-collaboration/spectre`
  `develop`. Lane worktrees share refs with the main checkout, so the
  push works from the user's own shell without entering scratch:
  `git -C /users/nilsvu/spectre push origin fp/<slug>`. A PR entry is
  staged only after the co-review with the user is done; `body.md`'s
  testing section records what was run.
- Entries must be self-contained: a teammate (or `sxs-bot`, later)
  could post them without any agent context.

## Posting flow (user)

```sh
# from your own shell, in any checkout of the feature-parity branch:
bash support/FeatureParity/Outbox/NNN-<slug>/post.sh
```

Review the entry first; edit freely — the files are the proposal, not
a done deal. After posting, tell an agent (or run `/parity-outbox`):
it verifies the write landed via the read API, then removes the entry
directory in a commit whose message records the created URL. GitHub
holds the truth from that moment; the outbox drains to empty.
