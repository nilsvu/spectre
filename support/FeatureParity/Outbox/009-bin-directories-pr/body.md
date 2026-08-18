## Proposed changes

Give every simulation a run-local `bin` directory that holds everything a scheduled job needs after submission — the executables, the SpECTRE CLI and its Python package, the machine's environment script, and the submit script templates — and run from it instead of from the build directory.

Closes #7447

Today a queued or continuing job reaches back into the build directory at exactly the point where it can no longer be supervised. The generated `Submit.sh` invokes `${SPECTRE_CLI} resubmit` and `${SPECTRE_CLI} run-next` *after* the executable exits, with `SPECTRE_CLI` pointing into `<build_dir>/bin`, and three machine templates `source` an environment script out of the source tree. So recompiling, switching branches, or deleting the build directory changes or breaks the next segment and the next pipeline step — after the job has already spent its wallclock. The executable itself was already copied into the segments directory, so C++ recompiles were safe; the Python/CLI half was not.

The design and the settled open points are in #7447. This implements that scope: the update path (versioned bin directories) and static third-party linking are deferred to follow-up issues.

**What lands in the bin directory** (`create_bin_directory` in `support/Python/Schedule.py`):

- the executables — including those of later pipeline steps, so the handoff to them doesn't have to reach back into the build directory;
- `spectre` — a verbatim copy of the build directory's CLI, which needs no special handling because **a simulation mirrors the build directory's layout**. `cmake/SpectrePythonExecutable.sh` puts the `python` directory next to itself and the bootstrapped dependencies at their build-relative path on the `PYTHONPATH`, resolved from its own location, and inherits the environment's. One unconditional branch, no detection, no configure-time absolute paths. See "A simulation mirrors the build directory" below;
- `python/spectre/` — the Python package with its compiled bindings and the configured `Machine.yaml`;
- the bootstrapped third-party Python packages, when the build has any (`BOOTSTRAP_PY_DEPS`) — copied **next to** the bin directory rather than into it, at the same path relative to the simulation as they have relative to the build directory (usually `lib/pythonX.Y/site-packages`). Packages provided by the machine environment are not copied;
- `Env.sh` — the machine's environment script, newly configured into the Python package by CMake so the scheduler can copy it without touching the source tree. `SubmitTemplateBase.sh` sources it when it exists, so **every** machine template inherits the behaviour, not only the three that used to source the source tree. (The issue body wrote this file as `bin/env.sh`; the final spelling is `bin/Env.sh`, matching the `support/Machines/<Machine>/Env.sh` layout that #7443 proposes);
- the submit script template and its base — now the one copy the scheduler renders from;
- `Manifest.yaml` — build directory, source revision (read from `BuildInfo.txt`), and timestamp.

**Almost no plumbing is needed for later segments, because the CLI already self-locates**: `spectre.__main__`, `_resolve_executable` and `spectre_cli` are all computed relative to `__file__`. Once the bin directory's CLI is the one running, everything resolves out of it by itself. Only the first `schedule` call — the one that runs from the build directory and creates the bin directory — points `spectre_cli` and the executable paths into it. `Resubmit.py` is unchanged.

#### One bin directory per simulation, shared by its branches

Eccentricity control branches resolutions by scheduling into numbered subdirectories of the simulation (`support/Pipelines/Bbh/EccentricityControl.py:138` passes `pipeline_dir=lev_dir.path`). Deriving the location from the target directory alone would give every Lev its own copy — several times 497 MB for one simulation. #5951 asked for the opposite ("one bin directory for all Ecc iterations, levs, segments"), and settled open point 2(a) quotes it.

So `schedule` does not just derive a location, it *looks one up*: `find_bin_directory` checks the target directory for a `bin`, then the enclosing directories, and the first hit is the simulation's bin directory, reused as is. A new one is created only when nothing encloses the run. A candidate counts only if it holds a `Manifest.yaml`, so a build directory's `bin` is never mistaken for a simulation's.

The search is bounded **structurally**, not by depth: it ascends out of a directory only while that directory is one the scheduler itself creates — a `Segment` or a `PipelineStep`, matched with the classes in `support/Python/DirectoryStructure.py` rather than re-spelled patterns. The first directory that is neither is the simulation root: it is checked, and the search stops without ever inspecting its parents. A run tree placed below an unrelated simulation therefore cannot pick up that simulation's bin directory — there is a test that plants a decoy one level above a non-conforming directory and asserts it is not found.

This only discovers existing bin directories. `create_bin` still decides whether one is *created*, and still travels through `Next` for the reason below.

The bin directory is created once per simulation and never updated implicitly: an executable that is already there is kept rather than replaced, and `--force` does not override that — replacing the executable of a running simulation is not something a scheduling flag should do. It goes into the pipeline directory when there is one, so all steps of a pipeline share it, otherwise into the segments directory, otherwise into the run directory. Runs without a scheduler (`--no-schedule`) never create or use one, since nothing runs unsupervised after you start them.

A bin directory is sizeable: **497 MB measured** for one simulation in a `Debug` build. Almost all of it is the copied Python package (533 MB in the build directory before `__pycache__` is dropped) — the compiled bindings dominate and the executables are a small part. A `Release` build with stripped symbols is far smaller. This is the disk concern raised in #5951; it is one copy per simulation, and eccentricity control gives each Lev branch its own pipeline directory and therefore its own copy.

### Upgrade instructions

<!-- UPGRADE INSTRUCTIONS -->
**`--copy-executable` / `--no-copy-executable` is removed.** Use `--create-bin` / `--no-create-bin` instead (`copy_executable` becomes `create_bin` in Python). There is no deprecated alias: passing the old flag on the command line now errors with "no such option". Update any scripts, notebooks or wrappers that use it. The `Next` blocks in the input files shipped with SpECTRE are updated in this PR; hand-written ones that pass `copy_executable` must be updated too.

Scheduler context files written before this change contain a `copy_executable` key. It is ignored, so **existing runs continue to resubmit** — there is a test for this. Their next segment gets a bin directory that keeps the executable **the run was created with**, not whatever is in the build directory now.

**The executable and the submit script templates move** from `<segments_dir>/` into `<segments_dir>/bin/`. Anything that hard-codes those paths needs updating.

**Builds configured with `BUILD_SHARED_LIBS=ON` can no longer schedule runs** unless you pass `--no-create-bin`. Creating a bin directory deliberately fails when an executable loads shared libraries out of the build directory, because such a copy breaks as soon as the build directory changes (settled open point 6(a) of #7447). Four of the repo's own environment scripts configure exactly that: `support/Environments/urania.sh:53`, `support/Environments/viper.sh:55`, `support/Environments/ocean2.sh:80` and `support/Environments/ocean2_orca1.sh:63`. On Urania, Viper, Ocean2 and Ocean2_orca1 a standard build therefore hits the guard. See "Further comments" for why that is not resolved here.

**Bootstrapped Python dependencies are now found where pip actually put them.** With `BOOTSTRAP_PY_DEPS` / `SPECTRE_FETCH_MISSING_DEPS` enabled, the build no longer assumes `lib/pythonX.Y/site-packages` but discovers the directory. On Debian-patched pips that is `local/lib/pythonX.Y/dist-packages`, which the build never put on the `PYTHONPATH`, so bootstrapped dependencies were silently unusable there. If your build tree has stale copies in more than one of those locations, configuring now stops with an error listing them — delete the stale ones.

**`<build_dir>/bin/spectre` and `bin/python-spectre` changed shape.** They now derive the `PYTHONPATH` from their own location — the `python` directory next to the script, and the bootstrapped dependencies at their build-relative path — and **append the environment's `PYTHONPATH` instead of baking in the one from configure time**. A shell that relied on the wrapper carrying the configure-time `PYTHONPATH` (including the SpEC paths that `SPEC_ROOT` adds) must now load that environment itself; in scheduled jobs the submit script's `Env.sh` sourcing does it. Anything that parses or patches these generated scripts should be checked. Their `configure_file` calls gained `@ONLY`, because the scripts now contain shell `${...}` expansions that CMake must not substitute.

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

- **New directories** in the pipeline or segments directory: `bin/`, plus a copy of the bootstrapped Python dependencies at their build-relative path (e.g. `lib/pythonX.Y/site-packages`) when the build has any. The job's log header prints the bin directory.
- **New context entries** `bin_dir` and `create_bin` in `SchedulerContext.yaml`. Only `create_bin` — the intent, not a path — is propagated through pipeline `Next` blocks; the location is derived from the pipeline directory, which already flows through `Next`, and is recorded in the context file. Threading a path through input files would be redundant state.
- **Submit scripts**: `SubmitTemplateBase.sh` sources `<bin_dir>/Env.sh` if it exists, so any machine gets its environment frozen. `Urania.sh`, `Deucalion.sh` and `Viper.sh` only had to stop sourcing the source tree when a bin directory is present — two lines each. (`Anvil.sh`, `Expanse.sh` and `Frontera.sh` also source the source tree but are standalone hand-edited scripts, not scheduler templates, so they are untouched.)
- **Which environment script a machine gets** is found by lowercasing the machine name (`Urania` → `support/Environments/urania.sh`), which covers 9 of the 10 machines in `support/Machines/`. `CaltechHpc` is the exception (`caltech_hpc_gcc.sh`) and so gets no `Env.sh` — no regression, since its submit script sources nothing today either. This is a deliberately minimal mechanism that avoids a hand-maintained mapping table; #7443's machine-directory restructure is the natural place to give this a proper home, and would remove the naming convention.
- **Earlier failure for incomplete builds**: starting a pipeline that will continue into a later step now requires that step's executable to be compiled, because it is copied up front. Previously the pipeline failed at the handoff — after the earlier step had already run.
- **Docs**: a "Bin directories" section in `docs/Tutorials/Cli.md`.

#### A simulation mirrors the build directory

The wrapper used to need a mode switch because a simulation and the build directory were shaped differently. Making the shapes the same removes the problem instead of detecting it: a simulation gets `bin/` **plus** `lib/...` when the build bootstrapped Python dependencies, exactly where the build directory keeps them. The wrapper is then one unconditional branch — the package next to it, the dependencies at their build-relative path below the enclosing prefix — and `test_cli_finds_the_package_next_to_itself` checks it resolves correctly in the build tree, in a simulation with dependencies, and in one without.

**Fixed on the way: `BOOTSTRAP_PY_DEPS` dependencies were invisible on Debian-patched pips.** The build assumed `pip install --prefix P` lands in `P/lib/pythonX.Y/site-packages` and put that on the `PYTHONPATH`. Measured on this machine, that pip installs to `P/local/lib/python3.10/dist-packages` instead, so bootstrapped dependencies were downloaded and then never found. `BootstrapPyDeps.cmake` now *discovers* the directory pip actually used — globbing `<build>/{local/,}lib/python*/{site,dist}-packages` for a non-empty one, and failing with a descriptive error if several match — and that directory is what goes on the `PYTHONPATH`, into the wrapper, and into a simulation. This is a pre-existing bug, fixed here because the mirroring depends on knowing the real path.

pip stays on `--prefix`, deliberately: it consults the `PYTHONPATH`, so packages the machine already provides are skipped and only the missing ones are downloaded. `--target` would give a fixed layout but has no such check — measured, with `click` and `humanize` importable and on the `PYTHONPATH`, `--target` re-downloaded and installed both, while `--prefix` reported them "already satisfied". A stamp over the hashes of the requirement files additionally skips the pip invocation entirely while the requirements are unchanged.

Consumers of the composed `PYTHONPATH`, all of which follow the discovered path: `bin/LoadPython.sh`, the Python test environment used by `spectre_add_python_test` and `spectre_add_python_bindings_test`, the input-file test environment (`AddInputFileTests.cmake`), the Catch test environment (`SpectreAddCatchTests.cmake`), and the configure-time module probes (`FindPythonModule.cmake`, `FindNumPy.cmake`). `SetupPybind11.cmake` used the assumed site-packages path as a `find_package` hint and now uses the discovered one — pybind11 is in `requirements.txt`, so it could be bootstrapped. `SetupSpec.cmake` still prepends SpEC's Python directories under `SPEC_ROOT`. Nothing else referenced the old path: the two `site-packages` mentions in `.github/workflows/Tests.yaml` are ParaView's and the CI interpreter's own, the install rule in `SpectreSetupPythonPackage.cmake` targets `CMAKE_INSTALL_PREFIX`, and `docs/Tutorials/Python.md` only ever named `BUILD_DIR/bin/python`.

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
# (support.Python.Schedule: 10 test cases)
```

`tests/support/Python/Test_Schedule.py` gains seven test cases. `test_bin_directory` covers:

- the contents of the bin directory, including that `env.sh` is present exactly when the build has a machine environment script, and that the executable and submit script templates are no longer in the segments directory;
- that neither the rendered `Submit.sh` nor `SchedulerContext.yaml` contains any path under the build directory's `bin` or `lib`, or under the source tree — the property this issue is about;
- that scheduling again reuses the bin directory instead of copying over it;
- that `<bin_dir>/spectre resubmit` works as a subprocess **with `PATH` and `PYTHONPATH` scrubbed**, so the build directory is unreachable through the environment, and writes bin-directory paths into the next segment. This exercises the relocatable wrapper and proves the copied package stands alone.

The other three:

- `test_no_bin_directory_without_scheduler` — a directly executed run copies nothing, even with `create_bin=True`;
- `test_relocatable_executable_guard` — the guard raises with the offending library path in the message and leaves no half-created directory (the library listing is stubbed, since triggering it for real needs a `BUILD_SHARED_LIBS=ON` build);
- `test_bin_directory_shared_by_branches` — a Lev branch nested in the simulation reuses the simulation's bin directory instead of creating its own, and its submit script points there; with no enclosing bin directory an opt-out still creates nothing;
- `test_bin_directory_search_stays_in_the_simulation` — a decoy `bin/Manifest.yaml` above a non-conforming directory is not picked up, while the same tree does find its own simulation's bin directory;
- `test_pipeline_steps_share_bin_directory` — two steps of a pipeline share the one bin directory: the second reuses it instead of re-copying, and its submit script points there. It also covers the opt-out surviving the handoff. This one caught a real bug: a later step passing its own submit script template raised `OSError: File already exists` in the bin directory, because only the executables kept the "already there wins" rule. Copying into the bin directory now goes through one helper that applies that rule to the templates as well;
- `test_cli_finds_the_package_next_to_itself` — the wrapper resolves to exactly the package directory next to it plus the bootstrapped dependencies at their build-relative path, checked in the build tree, in a simulation that has dependencies mirrored into it, and in one that does not;
- `test_submit_template_env_script` — renders **every** Jinja submit template in `support/SubmitScripts/` with and without a bin directory, checking that each switches to `bin/env.sh` and is otherwise unchanged from before. A build without a `MACHINE` never renders these, so they would have no coverage otherwise.

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

The build used for these tests has every Python dependency already provided by the machine, so pip installs nothing and there is no bootstrapped directory to mirror; the mirroring is covered by the unit test instead, which builds the layouts directly.

Two pre-existing build failures blocked `all-pybindings` here, neither related to this change: `PyCoordinateMaps` fails to compile under `nvcc` ("identifier ... is undefined in device code"), and `PySpectral` fails to link with `cannot find -lxsimd` because `src/Utilities/Simd/CMakeLists.txt` links the `xsimd` target unconditionally, so `USE_XSIMD=OFF` produces an unlinkable build. Both are worth separate issues.

---

🤖 drafted by: [Claude Fable 5](https://claude.com/claude-code)
(and reviewed by me 🙋)
