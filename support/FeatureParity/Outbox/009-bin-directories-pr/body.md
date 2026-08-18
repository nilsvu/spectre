## Proposed changes

Give every simulation a run-local `bin` directory that holds everything a scheduled job needs after submission — the executables, the SpECTRE CLI and its Python package, the machine's environment script, and the submit script templates — and run from it instead of from the build directory.

Closes #7447

Today a queued or continuing job reaches back into the build directory at exactly the point where it can no longer be supervised. The generated `Submit.sh` invokes `${SPECTRE_CLI} resubmit` and `${SPECTRE_CLI} run-next` *after* the executable exits, with `SPECTRE_CLI` pointing into `<build_dir>/bin`, and three machine templates `source` an environment script out of the source tree. So recompiling, switching branches, or deleting the build directory changes or breaks the next segment and the next pipeline step — after the job has already spent its wallclock. The executable itself was already copied into the segments directory, so C++ recompiles were safe; the Python/CLI half was not.

The design and the settled open points are in #7447. This implements that scope: the update path (versioned bin directories) and static third-party linking are deferred to follow-up issues.

**What lands in the bin directory** (`create_bin_directory` in `support/Python/Schedule.py`):

- the executables — including those of later pipeline steps, so the handoff to them doesn't have to reach back into the build directory;
- `spectre` — a **relocatable** CLI wrapper. The build directory's wrapper can't be copied verbatim because CMake bakes an absolute `PYTHONPATH` into it; this one resolves the package relative to its own location and appends the environment's `PYTHONPATH`;
- `python/spectre/` — the Python package with its compiled bindings and the configured `Machine.yaml`;
- `python-deps/` — third-party Python packages that CMake bootstrapped into the build directory (`BOOTSTRAP_PY_DEPS`), when there are any. Packages provided by the machine environment are not copied;
- `env.sh` — the machine's environment script, newly configured into the Python package by CMake so the scheduler can copy it without touching the source tree;
- the submit script template and its base — now the one copy the scheduler renders from;
- `Manifest.yaml` — build directory, source revision (read from `BuildInfo.txt`), and timestamp.

**Almost no plumbing is needed for later segments, because the CLI already self-locates**: `spectre.__main__`, `_resolve_executable` and `spectre_cli` are all computed relative to `__file__`. Once the bin directory's CLI is the one running, everything resolves out of it by itself. Only the first `schedule` call — the one that runs from the build directory and creates the bin directory — points `spectre_cli` and the executable paths into it. `Resubmit.py` is unchanged.

The bin directory is created once per simulation and never updated implicitly: an executable that is already there is kept rather than replaced, and `--force` does not override that — replacing the executable of a running simulation is not something a scheduling flag should do. It goes into the pipeline directory when there is one, so all steps of a pipeline share it, otherwise into the segments directory, otherwise into the run directory. Runs without a scheduler (`--no-schedule`) never create or use one, since nothing runs unsupervised after you start them.

A bin directory is sizeable: **497 MB measured** for one simulation in a `Debug` build. Almost all of it is the copied Python package (533 MB in the build directory before `__pycache__` is dropped) — the compiled bindings dominate and the executables are a small part. A `Release` build with stripped symbols is far smaller. This is the disk concern raised in #5951; it is one copy per simulation, and eccentricity control gives each Lev branch its own pipeline directory and therefore its own copy.

### Upgrade instructions

<!-- UPGRADE INSTRUCTIONS -->
**`--copy-executable` / `--no-copy-executable` is removed.** Use `--create-bin` / `--no-create-bin` instead (`copy_executable` becomes `create_bin` in Python). There is no deprecated alias: passing the old flag on the command line now errors with "no such option". Update any scripts, notebooks or wrappers that use it. The `Next` blocks in the input files shipped with SpECTRE are updated in this PR; hand-written ones that pass `copy_executable` must be updated too.

Scheduler context files written before this change contain a `copy_executable` key. It is ignored, so **existing runs continue to resubmit** — there is a test for this. Their next segment gets a bin directory that keeps the executable **the run was created with**, not whatever is in the build directory now.

**The executable and the submit script templates move** from `<segments_dir>/` into `<segments_dir>/bin/`. Anything that hard-codes those paths needs updating.

**Builds configured with `BUILD_SHARED_LIBS=ON` can no longer schedule runs** unless you pass `--no-create-bin`. Creating a bin directory deliberately fails when an executable loads shared libraries out of the build directory, because such a copy breaks as soon as the build directory changes (settled open point 6(a) of #7447). Four of the repo's own environment scripts configure exactly that: `support/Environments/urania.sh:53`, `support/Environments/viper.sh:55`, `support/Environments/ocean2.sh:80` and `support/Environments/ocean2_orca1.sh:63`. On Urania, Viper, Ocean2 and Ocean2_orca1 a standard build therefore hits the guard. See "Further comments" for why that is not resolved here.

Pass `--no-create-bin` for quick tests where the copy isn't worth it. Note that it includes the compiled Python bindings, so it is sizeable in `Debug` builds.
<!-- UPGRADE INSTRUCTIONS -->

### Code review checklist

- [ ] The code is documented and the documentation renders correctly. Run
  `make doc` to generate the documentation locally into `BUILD_DIR/docs/html`.
  Then open `index.html`.
- [ ] The code follows the stylistic and code quality guidelines listed in the
  [code review guide](https://spectre-code.org/code_review_guide.html).
- [ ] The PR lists upgrade instructions and is labeled `bugfix` or
  `new feature` if appropriate.
- [ ] If a coding agent was used, have a co-author trailer of the form
      "Co-Authored-By: <agent name> <agent email>" as the last line of the
      commit, e.g. "Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>",
      "Co-Authored-by: Codex <noreply@openai.com>", or
      "Co-Authored-By: GitHub Copilot CLI <noreply@microsoft.com>".

### Further comments

#### The shared-libraries guard collides with four of our own machine setups

This is the one thing here that needs a decision from someone other than me. The guard is the settled design and it works — but `support/Environments/{urania,viper,ocean2,ocean2_orca1}.sh` all configure `-D BUILD_SHARED_LIBS=ON`, so on those four machines a standard build will refuse to create a bin directory and `spectre schedule` will fail with the guard's message.

That flag choice predates this PR and carries no recorded rationale in the repo. Resolving it — flip those four scripts to static linking, or keep shared libraries there and have those machines pass `--no-create-bin` — is deliberately **not** decided here: it belongs to the machine-setup issue #7443 (its open point 8). No environment script is touched in this PR.

The guard covers the executables only, not the compiled Python bindings copied into `bin/python/`. In a shared-libraries build those `.so` files link against the build tree too, so such a bin directory would be broken in a second way — but the executable check fires first and stops it, so that path is unreachable in practice. Extending the guard to the bindings is deliberately out of scope.

#### Other user-visible effects

- **New directory** `bin/` in the pipeline or segments directory. The job's log header prints it.
- **New context entries** `bin_dir` and `create_bin` in `SchedulerContext.yaml`. Only `create_bin` — the intent, not a path — is propagated through pipeline `Next` blocks; the location is derived from the pipeline directory, which already flows through `Next`, and is recorded in the context file. Threading a path through input files would be redundant state.
- **Submit scripts** for Urania, Deucalion and Viper source `bin/env.sh` when the run has a bin directory, and keep sourcing the source tree otherwise. (`Anvil.sh`, `Expanse.sh` and `Frontera.sh` also source the source tree but are standalone hand-edited scripts, not scheduler templates, so they are untouched.)
- **Earlier failure for incomplete builds**: starting a pipeline that will continue into a later step now requires that step's executable to be compiled, because it is copied up front. Previously the pipeline failed at the handoff — after the earlier step had already run.
- **Docs**: a "Bin directories" section in `docs/Tutorials/Cli.md`.

#### Known limitation

Moving or renaming a simulation directory still fails on resubmission, because the executable path recorded in `SchedulerContext.yaml` is absolute and stale, and `_resolve_executable` rejects it before the bin directory is consulted. The bin directory itself recovers — a stale recorded `bin_dir` falls back to the CLI's own location when that is a bin directory — so only the executable path is left. Resolving a stale executable by name would fix it; that is not attempted here.

### Testing performed

This change is pipeline-side only — no `src/` code is touched, so a full `ctest -L unit` run is not informative for it and was not run. The affected Python and support tests were run in a `Debug` build:

```sh
ctest -R "support\.(Python\.(Schedule|Main|RunNext)|DirectoryStructure|Machines)" --output-on-failure
# 100% tests passed, 0 tests failed out of 5
# (support.Python.Schedule: 7 test cases, 2.2 s)
```

`tests/support/Python/Test_Schedule.py` gains four test cases. `test_bin_directory` covers:

- the contents of the bin directory, including that `env.sh` is present exactly when the build has a machine environment script, and that the executable and submit script templates are no longer in the segments directory;
- that neither the rendered `Submit.sh` nor `SchedulerContext.yaml` contains any path under the build directory's `bin` or `lib`, or under the source tree — the property this issue is about;
- that scheduling again reuses the bin directory instead of copying over it;
- that `<bin_dir>/spectre resubmit` works as a subprocess **with `PATH` and `PYTHONPATH` scrubbed**, so the build directory is unreachable through the environment, and writes bin-directory paths into the next segment. This exercises the relocatable wrapper and proves the copied package stands alone.

The other three:

- `test_no_bin_directory_without_scheduler` — a directly executed run copies nothing, even with `create_bin=True`;
- `test_relocatable_executable_guard` — the guard raises with the offending library path in the message and leaves no half-created directory (the library listing is stubbed, since triggering it for real needs a `BUILD_SHARED_LIBS=ON` build);
- `test_submit_template_env_script` — renders the real `SubmitTemplateBase.sh`, `Urania.sh`, `Deucalion.sh` and `Viper.sh` with and without a bin directory, checking that they switch to `bin/env.sh` and are otherwise unchanged from before. A build without a `MACHINE` never renders these, so they would have no coverage otherwise.

`test_schedule` now covers the `--no-create-bin` layout, which is also the layout of runs scheduled before this change, and ends by resubmitting a simulated pre-change `SchedulerContext.yaml` (no `create_bin`, obsolete `copy_executable` key, executable in the segments directory) to check that old runs keep working.

The Bbh pipeline tests pass `--no-create-bin` so they keep testing pipeline wiring without copying the Python package on every invocation; their expected `Next` blocks were updated for the renamed option. **These tests were not executed** — the pybindings they need don't link in the environment this was developed in, for reasons unrelated to this change (below). Their expectations were instead checked by rendering the pipeline templates directly and confirming that `create_bin` reaches the `Next` block as a boolean. **Please make sure CI runs `support.Pipelines.Bbh.*`.**

Build-directory independence was also checked by hand, outside the test suite and outside the build directory: schedule a run, move the build directory away entirely, and resubmit through the bin directory's CLI.

```sh
mv $BUILD_DIR ${BUILD_DIR}-MOVED
$RUN_DIR/Segments/bin/spectre resubmit $RUN_DIR/Segments --submit
```

It succeeded (exit 0), created `Segment_0001`, and the new submit script references only the bin directory:

```sh
SPECTRE_EXECUTABLE=.../Segments/bin/TestExec
SPECTRE_CLI=.../Segments/bin/spectre
```

The `python-deps/` path was exercised separately by placing a package in `<build_dir>/lib/pythonX.Y/site-packages` and confirming it lands in the bin directory; the build used for these tests bootstraps no Python dependencies of its own.

Two pre-existing build failures blocked `all-pybindings` here, neither related to this change: `PyCoordinateMaps` fails to compile under `nvcc` ("identifier ... is undefined in device code"), and `PySpectral` fails to link with `cannot find -lxsimd` because `src/Utilities/Simd/CMakeLists.txt` links the `xsimd` target unconditionally, so `USE_XSIMD=OFF` produces an unlinkable build. Both are worth separate issues.

---

🤖 drafted by: [Claude Fable 5](https://claude.com/claude-code)
(and reviewed by me 🙋)
