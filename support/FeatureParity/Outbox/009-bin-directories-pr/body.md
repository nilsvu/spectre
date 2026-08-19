## Proposed changes

Give every simulation a run-local `bin` directory that holds everything a scheduled job needs after submission — the executables, the SpECTRE CLI and its Python package, and the submit script templates — and run from it instead of from the build directory.

Closes #7447

Today a queued or continuing job reaches back into the build directory at exactly the point where it can no longer be supervised. The generated `Submit.sh` invokes `${SPECTRE_CLI} resubmit` and `${SPECTRE_CLI} run-next` *after* the executable exits, with `SPECTRE_CLI` pointing into `<build_dir>/bin`, and three machine templates `source` an environment script out of the source tree. So recompiling, switching branches, or deleting the build directory changes or breaks the next segment and the next pipeline step — after the job has already spent its wallclock. The executable itself was already copied into the segments directory, so C++ recompiles were safe; the Python/CLI half was not.

The design and the settled open points are in #7447. This implements that scope: the update path (versioned bin directories) and static third-party linking are deferred to follow-up issues.

The bin directory lives in its own module, `support/Python/BinDirectory.py` — it is a self-contained concern (find it, create it, add to it) that `Schedule.py` only calls into, and the deferred `update-bin` endpoint will land there too. Imports run one way: `Schedule` imports from `BinDirectory`, never the reverse.

**What lands in the bin directory** (`create_bin_directory`):

- the executables — including those of later pipeline steps, so the handoff to them doesn't have to reach back into the build directory. Callers name them: `schedule` takes a `copy_extra_executables` list, and the Bbh pipeline lists the literal names. It deliberately does **not** read them out of the later steps' input file templates, because the executable a step will actually run is the caller's choice — the GPU build substitutes its own — so the template's name would be wrong by design;
- `spectre` — a verbatim copy of the build directory's CLI. `cmake/SpectrePythonExecutable.sh` reaches the Python package through its own location, so the copy uses the package next to it; everything else on its `PYTHONPATH` stays exactly as configured at build time. That is a two-line change to the wrapper and needs no detection of where it is running. See "One self-locating entry on the `PYTHONPATH`" below;
- `python/spectre/` — the Python package with its compiled bindings and the configured `Machine.yaml`;
- the submit script template and its base — the one copy the scheduler renders from. The templates follow the bin directory: a run without one renders them where they are and copies nothing.

The bin directory carries no record file: formaline already embeds the source archive and environment in every executable and H5 output.

**Almost no plumbing is needed for later segments, because the CLI already self-locates**: `spectre.__main__`, `_resolve_executable` and `spectre_cli` are all computed relative to `__file__`. Once the bin directory's CLI is the one running, everything resolves out of it by itself. Only the first `schedule` call — the one that runs from the build directory and creates the bin directory — points `spectre_cli` and the executable paths into it. `Resubmit.py` is unchanged.

#### One bin directory per simulation, shared by its branches

Eccentricity control branches resolutions by scheduling into numbered subdirectories of the simulation (`support/Pipelines/Bbh/EccentricityControl.py:138` passes `pipeline_dir=lev_dir.path`). Deriving the location from the target directory alone would give every Lev its own copy — several times 497 MB for one simulation. #5951 asked for the opposite ("one bin directory for all Ecc iterations, levs, segments"), and settled open point 2(a) quotes it.

So `schedule` does not just derive a location, it *looks one up*: `find_bin_directory` starts at the **run directory** — the most specific directory of the run — and works outwards, and the first hit is the simulation's bin directory, reused as is. Starting at the most specific directory means the nearest bin directory wins, so a run that already has one of its own is not silently shadowed by a more distant one. That layout arises when a run scheduled on its own with `--segments-dir` is later made part of a pipeline: the pipeline directory encloses a segments directory that already has a bin directory, and the segments keep using it. A new one is created only when the search finds nothing, and then it is created for the simulation as a whole — in the pipeline directory if there is one, otherwise the segments directory, otherwise the run directory. A candidate counts only if it contains the copied `spectre` CLI and does not sit in a build directory (no `CMakeCache.txt` next to it), so a build directory's `bin` is never mistaken for a simulation's.

The search is bounded **structurally**, not by depth: it ascends out of a directory only while that directory is one the scheduler itself creates — a `Segment` or a `PipelineStep`, matched with the classes in `support/Python/DirectoryStructure.py` rather than re-spelled patterns. The first directory that is neither is the simulation root: it is checked, and the search stops without ever inspecting its parents. A run tree placed below an unrelated simulation therefore cannot pick up that simulation's bin directory — there is a test that plants a decoy one level above a non-conforming directory and asserts it is not found.

This only discovers existing bin directories. `create_bin` still decides whether one is *created*, and still travels through `Next` for the reason below.

The bin directory is created once per simulation and never updated implicitly: an executable that is already there is kept rather than replaced, and `--force` does not override that — replacing the executable of a running simulation is not something a scheduling flag should do. It goes into the pipeline directory when there is one, so all steps of a pipeline share it, otherwise into the segments directory, otherwise into the run directory. Runs without a scheduler (`--no-schedule`) never create or use one, since nothing runs unsupervised after you start them.

A bin directory is sizeable: **497 MB measured** for one simulation in a `Debug` build. That also makes `support.Python.Schedule` an I/O-heavy test — it creates three of them — which is why it carries a longer timeout than its neighbours. Almost all of it is the copied Python package (533 MB in the build directory before `__pycache__` is dropped) — the compiled bindings dominate and the executables are a small part. A `Release` build with stripped symbols is far smaller. This is the disk concern raised in #5951; it is one copy per simulation, and eccentricity control gives each Lev branch its own pipeline directory and therefore its own copy.

### Upgrade instructions

<!-- UPGRADE INSTRUCTIONS -->
**`--copy-executable` / `--no-copy-executable` is removed.** Use `--create-bin` / `--no-create-bin` instead (`copy_executable` becomes `create_bin` in Python). There is no deprecated alias: passing the old flag on the command line now errors with "no such option". Update any scripts, notebooks or wrappers that use it. The `Next` blocks in the input files shipped with SpECTRE are updated in this PR; hand-written ones that pass `copy_executable` must be updated too.

Scheduler context files written before this change contain a `copy_executable` key. It is ignored, so **existing runs continue to resubmit** — there is a test for this. Their next segment gets a bin directory that keeps the executable **the run was created with**, not whatever is in the build directory now.

**The executable and the submit script templates move** from `<segments_dir>/` into `<segments_dir>/bin/`. Anything that hard-codes those paths needs updating. With `--no-create-bin` the templates are no longer copied into the segments directory at all — the run renders them where they are, as a plain `--run-dir` run always did. Templates supplied with `--submit-script-template` behave the same way.

**A run without a scheduler now uses an existing bin directory.** `--no-schedule` never *creates* one — not even with `--create-bin` — but if the simulation already has one, the run executes the copy in it rather than the build directory's. So running a segment by hand inside a simulation runs the same binary its scheduled jobs do. Previously a direct run ignored the bin directory entirely.

**Builds configured with `BUILD_SHARED_LIBS=ON` can no longer schedule runs** unless you pass `--no-create-bin`. Creating a bin directory deliberately fails when the build directory holds shared SpECTRE libraries, because its executables load them from there and a copy breaks as soon as the build directory changes (settled open point 6(a) of #7447). Four of the repo's own environment scripts configure exactly that: `support/Environments/urania.sh:53`, `support/Environments/viper.sh:55`, `support/Environments/ocean2.sh:80` and `support/Environments/ocean2_orca1.sh:63`. On Urania, Viper, Ocean2 and Ocean2_orca1 a standard build therefore hits the guard. See "Further comments" for why that is not resolved here.

**`<build_dir>/bin/spectre` and `bin/python-spectre` reach the Python package through their own location** instead of an absolute configure-time path. Everything else on their `PYTHONPATH` is unchanged. In the build directory they behave exactly as before; the point is that a copy of the script in a simulation's bin directory uses the package next to the copy. Their `configure_file` calls gained `@ONLY`, because the scripts now contain a shell `${...}` expansion that CMake must not substitute.

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

#### Freezing the machine environment is deferred to #7443

Settled open point 5(a) asked for the machine's environment script to be copied into the bin directory and sourced from there, so that a job loads the environment the simulation was created with. That was implemented and then withdrawn: it needed CMake to work out which script belongs to a machine, which is a mapping that #7443's machine-directory design owns. Submit scripts therefore keep today's behaviour exactly — the machines that source an environment script still source it from the source tree — and the freezing lands with #7443. Nothing in `support/SubmitScripts/` changes here except the bin directory line in the job's log header.

The guard detects the build shape rather than inspecting each executable: it refuses when `<build_dir>/lib` holds shared objects, which is exactly what `BUILD_SHARED_LIBS=ON` produces there. That assumes shared libraries directly in `lib` are SpECTRE's; the subdirectories where bootstrapped Python packages keep their own `.so` files are not looked at. It is a coarser test than reading each executable's `NEEDED` entries, and deliberately so — it needs no `ldd` subprocess and no new configuration to interrogate, and the condition it detects is the one that matters.

#### Other user-visible effects

- **New directory** `bin/` in the pipeline or segments directory. The job's log header prints it.
- **New context entries** `bin_dir` and `create_bin` in `SchedulerContext.yaml`. Only `create_bin` — the intent, not a path — is propagated through pipeline `Next` blocks; the location is derived from the pipeline directory, which already flows through `Next`, and is recorded in the context file. Threading a path through input files would be redundant state.
- **Earlier failure for incomplete builds**: starting a pipeline that will continue into a later step now requires that step's executable to be compiled, because it is copied up front. Previously the pipeline failed at the handoff — after the earlier step had already run.
- **Docs**: a "Bin directories" section in `docs/Tutorials/Cli.md`.

#### One self-locating entry on the `PYTHONPATH`

The wrapper needs exactly one change to work in a simulation: reach the Python package through its own location rather than through the absolute path configured into it. Everything after that first entry stays as the build configured it, so a copied wrapper carries the build directory's paths as trailing entries. **They are not removed, deliberately** — the frozen package comes first and wins, and keeping the rest untouched makes the change two lines instead of a mode switch that has to detect where it is running. In the build directory the self-located entry duplicates the first configured one, and Python drops the duplicate when it builds `sys.path`.

**Dependency bootstrapping is untouched here.** A pre-existing defect — `BOOTSTRAP_PY_DEPS` dependencies are installed where nothing looks for them on Debian-patched pips — was found while developing this change and is filed as its own issue with the measurements.

**Third-party Python packages are not frozen into a simulation.** A scheduled run gets them from the machine's Python environment, the same way the build directory does. The evidence that this is what already happens: bootstrapping was silently broken on Debian pips and nobody noticed, and the production environment scripts supply a venv or Spack environment. Freezing them was implemented and then dropped as unnecessary weight; if a module upgrade ever breaks an old simulation, that is the same exposure open point 7 already records.

#### Why `create_bin` still travels through `Next`

It looks redundant — later pipeline steps find the pipeline-level `bin/` already there, so the create-once early return makes the flag a no-op. That is true for the normal path, and measured: with the first step creating the bin directory, a second step scheduled with the flag unset reuses it and does not re-copy (its `spectre` mtime is unchanged).

One case breaks it, also measured: if the first step ran with `--no-create-bin`, there is no bin directory for the later step to find, and the later step's default (`pipeline_dir` is set, so create one) turns the explicit opt-out into a 497 MB copy taken mid-pipeline from whatever build directory the handoff CLI happens to be running from. Passing the intent forward is what prevents that, so the propagation stays. The other cases are unaffected: standalone later steps and segments-only runs never go through a `Next` block, and input files that still carry `create_bin` keep working because it is a normal `schedule` argument.

#### Known limitation

Moving or renaming a simulation directory still fails on resubmission, because the executable path recorded in `SchedulerContext.yaml` is absolute and stale, and `_resolve_executable` rejects it before the bin directory is consulted. The bin directory itself recovers — a stale recorded `bin_dir` falls back to the CLI's own location when that is a bin directory — so only the executable path is left. Resolving a stale executable by name would fix it; that is not attempted here.

### Testing performed

This change is pipeline-side only — no `src/` code is touched, so a full `ctest -L unit` run is not informative for it and was not run. The affected Python and support tests were run in a `Debug` build:

```sh
ctest -R "support\.(Python\.(Schedule|Main|RunNext)|DirectoryStructure|Machines)" --output-on-failure
# 100% tests passed, 0 tests failed out of 5
# (support.Python.Schedule: 9 test cases)
```

The bin-directory cases live in a new `tests/support/Python/Test_BinDirectory.py` (5 cases); `Test_Schedule.py` keeps the scheduling, resubmission and CLI cases and goes back to 4. `test_bin_directory` covers:

- the contents of the bin directory, and that the executable and submit script templates are no longer in the segments directory;
- that neither the rendered `Submit.sh` nor `SchedulerContext.yaml` contains any path under the build directory's `bin` or `lib`, or under the source tree — the property this issue is about;
- that scheduling again reuses the bin directory instead of copying over it;
- that `<bin_dir>/spectre resubmit` works as a subprocess **with `PATH` and `PYTHONPATH` scrubbed**, so the build directory is unreachable through the environment, and writes bin-directory paths into the next segment. This exercises the relocatable wrapper and proves the copied package stands alone.

The other three:

- `test_no_bin_directory_without_scheduler` — a directly executed run copies nothing, even with `create_bin=True`;
- `test_relocatable_build_guard` — the guard raises on a planted `lib/libDataStructures.so` with the path in the message, accepts a static `.a` next to it, and is not tripped by shared objects in the subdirectories where bootstrapped Python packages keep theirs;
- `test_bin_directory_search_stays_in_the_simulation` — a decoy `bin/` above a non-conforming directory is not picked up, the same tree does find its own simulation's bin directory, and a `bin` with a `CMakeCache.txt` next to it (a build directory's) is never matched;
- `test_bin_directory_shared_by_the_simulation` — the steps of a pipeline and a branch nested inside the simulation all share the one bin directory: later ones reuse it instead of re-copying, and their submit scripts point there. It also covers an explicit opt-out surviving the handoff, and it caught a real bug: a later step passing its own submit script template raised `OSError: File already exists`, because only the executables had the "already there wins" rule. Copying into the bin directory now goes through one helper that applies it to the templates too;
- `test_cli_finds_the_package_next_to_itself` — running the wrapper in the build tree and running a copy of it elsewhere both put the package next to the script first on `sys.path`, with the configured entries following unchanged;

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

Two pre-existing build failures blocked `all-pybindings` here, neither related to this change: `PyCoordinateMaps` fails to compile under `nvcc` ("identifier ... is undefined in device code"), and `PySpectral` fails to link with `cannot find -lxsimd` because `src/Utilities/Simd/CMakeLists.txt` links the `xsimd` target unconditionally, so `USE_XSIMD=OFF` produces an unlinkable build. Both are worth separate issues.

---

🤖 drafted by: [Claude Fable 5](https://claude.com/claude-code)
(and reviewed by me 🙋)
