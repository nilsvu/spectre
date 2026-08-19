## Proposed changes

Give every simulation a run-local `bin` directory that holds everything a scheduled job needs after submission — the executables, the SpECTRE CLI and its Python package — with the support files next to it in `support/`, and run from it instead of from the build directory.

Closes #7447

Today a queued or continuing job reaches back into the build directory at exactly the point where it can no longer be supervised. The generated `Submit.sh` invokes `${SPECTRE_CLI} resubmit` and `${SPECTRE_CLI} run-next` *after* the executable exits, with `SPECTRE_CLI` pointing into `<build_dir>/bin`, and three machine templates `source` an environment script out of the source tree. So recompiling, switching branches, or deleting the build directory changes or breaks the next segment and the next pipeline step — after the job has already spent its wallclock. The executable itself was already copied into the segments directory, so C++ recompiles were safe; the Python/CLI half was not.

The design and the settled open points are in #7447. This implements that scope: the update path (versioned bin directories) and static third-party linking are deferred to follow-up issues.

The bin directory lives in its own module, `support/Python/BinDirectory.py` — it is a self-contained concern (find it, create it, add to it) that `Schedule.py` only calls into, and the deferred `update-bin` endpoint will land there too. Imports run one way: `Schedule` imports from `BinDirectory`, never the reverse.

It is a frozen dataclass in the style of `Segment`, `PipelineStep` and `Checkpoint` in `DirectoryStructure.py`: one `path` field, the rest of the layout derived from it (`spectre_cli`, `python_dir`, `support_dir`), classmethod constructors (`this()`, `find()`, `create()`) and two operations (`add()`, `executable()`). The name of the CLI is written once, in `spectre_cli`, and everything that looks for it or copies it goes through that. `this()` self-locates from `__file__`, the way `Machines.this_machine()` names the machine this code runs on, and `create()` takes the installation to copy from as an argument that defaults to it. `schedule` threads one handle instead of parallel paths, and converts to a plain path at the boundary where `SchedulerContext.yaml` is written.

**What lands in the bin directory** (`BinDirectory.create`):

- the executables — including those of later pipeline steps, so the handoff to them doesn't have to reach back into the build directory. Callers name them: `schedule` takes a `copy_extra_executables` list, and the Bbh pipeline lists the literal names. It deliberately does **not** read them out of the later steps' input file templates: explicit beats parsing, and which executable a step runs is the caller's choice, so a caller can substitute a different one;
- `spectre` — a verbatim copy of the build directory's CLI. `cmake/SpectrePythonExecutable.sh` reaches the Python package through its own location, so the copy uses the package next to it; everything else on its `PYTHONPATH` stays exactly as configured at build time. That is a two-line change to the wrapper and needs no detection of where it is running. See "One self-locating entry on the `PYTHONPATH`" below;
- `python/` — the Python package with its compiled bindings, copied whole, so the packaging and shell-completion files the build directory keeps next to it come along (4 kB, none of them referencing the build directory).

**Next to it, `support/`** holds the submit script templates and the configured `Machine.yaml`. CMake configures them into `<build_dir>/support/` too (`support/CMakeLists.txt`), so **the support files are at the same place relative to `bin/` in both** and one expression — `BinDirectory.this().support_dir` — finds them in either. The machine environment script of #7443 joins them there.

The name is the source tree's own: these files are configured from `support/Machines/` and `support/SubmitScripts/`, so `<build_dir>/support/` and `<simulation_dir>/support/` reuse vocabulary every SpECTRE developer already knows. FHS names like `share/` would only earn their keep in a shared install prefix, which neither a build directory nor a simulation is. The build side reuses the output directory CMake already has for `add_subdirectory(support)`, so nothing new appears there.

**They are copied by name, not as a whole directory.** `<build_dir>/support/` is also where CMake generates `CTestTestfile.cmake` and `cmake_install.cmake`, which carry absolute build-directory paths and must never reach a simulation. `BinDirectory.SUPPORT_FILES` is the allowlist and mirrors what `support/CMakeLists.txt` configures — a new support file, such as the machine environment script of #7443, is added in both places. In a build directory the support files sit beside CMake's, which nothing enumerates; in a simulation, `support/` holds exactly the support files.

The bin directory carries no record file: formaline already embeds the source archive and environment in every executable and H5 output.

**Almost no plumbing is needed for later segments, because the CLI already self-locates**: `spectre.__main__`, `_resolve_executable` and `spectre_cli` are all computed relative to `__file__`. Once the bin directory's CLI is the one running, everything resolves out of it by itself. Only the first `schedule` call — the one that runs from the build directory and creates the bin directory — points `spectre_cli` and the executable paths into it. `Resubmit.py` is unchanged.

#### One bin directory per simulation, shared by its branches

Eccentricity control branches resolutions by scheduling into numbered subdirectories of the simulation (`support/Pipelines/Bbh/EccentricityControl.py:138` passes `pipeline_dir=lev_dir.path`). Deriving the location from the target directory alone would give every Lev its own copy — several times 497 MB for one simulation. #5951 asked for the opposite ("one bin directory for all Ecc iterations, levs, segments"), and settled open point 2(a) quotes it.

So `schedule` does not just derive a location, it *looks one up*: `BinDirectory.find` starts at the **run directory** — the most specific directory of the run — and works outwards, and the first hit is the simulation's bin directory, reused as is. Starting at the most specific directory means the nearest bin directory wins, so a run that already has one of its own is not silently shadowed by a more distant one. That layout arises when a run scheduled on its own with `--segments-dir` is later made part of a pipeline: the pipeline directory encloses a segments directory that already has a bin directory, and the segments keep using it. A new one is created only when the search finds nothing, and then it is created for the simulation as a whole — in the pipeline directory if there is one, otherwise the segments directory, otherwise the run directory. A bin directory is any directory named `bin` that holds the `spectre` CLI. A build directory's `bin` holds one too, and is deliberately not excluded: if a run tree sits directly in a build directory, running from that `bin` is exactly what scheduling from there means, and nothing has to be copied for it at all because the executables, the CLI and the support files are all already in place.

The search is bounded **structurally**, not by depth: it ascends out of a directory only while that directory is one the scheduler itself creates — a `Segment` or a `PipelineStep`, matched with the classes in `support/Python/DirectoryStructure.py` rather than re-spelled patterns. The first directory that is neither is the simulation root: it is checked, and the search stops without ever inspecting its parents. A run tree placed below an unrelated simulation therefore cannot pick up that simulation's bin directory — there is a test that plants a decoy one level above a non-conforming directory and asserts it is not found.

This only discovers existing bin directories. `create_bin` still decides whether one is *created*, and still travels through `Next` for the reason below.

The bin directory is created once per simulation and never updated implicitly: an executable that is already there is kept rather than replaced, and `--force` does not override that — replacing the executable of a running simulation is not something a scheduling flag should do. It goes into the pipeline directory when there is one, so all steps of a pipeline share it, otherwise into the segments directory, otherwise into the run directory. Runs without a scheduler (`--no-schedule`) never create or use one, since nothing runs unsupervised after you start them.

A bin directory is sizeable: **497 MB measured** for one simulation in a `Debug` build. That also makes `support.Python.Schedule` an I/O-heavy test — it creates three of them — which is why it carries a longer timeout than its neighbours. Almost all of it is the copied Python package (533 MB in the build directory before `__pycache__` is dropped) — the compiled bindings dominate and the executables are a small part. A `Release` build with stripped symbols is far smaller. This is the disk concern raised in #5951; it is one copy per simulation, and eccentricity control gives each Lev branch its own pipeline directory and therefore its own copy.

### Upgrade instructions

<!-- UPGRADE INSTRUCTIONS -->
**`--copy-executable` / `--no-copy-executable` is removed.** Use `--create-bin` / `--no-create-bin` instead (`copy_executable` becomes `create_bin` in Python). There is no deprecated alias: passing the old flag on the command line now errors with "no such option". Update any scripts, notebooks or wrappers that use it. The `Next` blocks in the input files shipped with SpECTRE are updated in this PR; hand-written ones that pass `copy_executable` must be updated too.

Scheduler context files written before this change contain a `copy_executable` key. It is ignored, so **existing runs continue to resubmit** — there is a test for this. Their next segment gets a bin directory that keeps the executable **the run was created with**, not whatever is in the build directory now.

**The executable moves** from `<segments_dir>/` into `<segments_dir>/bin/`, and the submit script templates are no longer copied into the segments directory at all. Anything that hard-codes those paths needs updating.

**The submit script templates and `Machine.yaml` are configured to `<build_dir>/support/`** instead of into the Python package at `<build_dir>/bin/python/spectre/support/`. Anything that reads them from the old path needs updating; in-tree, `Schedule.default_submit_script_template` and `Machines.this_machine` both resolve them relative to the running CLI now. A simulation gets a copy of them at `<simulation_dir>/support/`, and `SchedulerContext.yaml` records that copy, so **to change how an existing simulation's later segments are submitted you edit `<simulation_dir>/support/SubmitTemplate.sh`** — measured, see Testing. A template supplied with `--submit-script-template` from anywhere else is rendered where it is and is not copied; it can still `{% extends %}` the installed base template, which the Jinja loader finds in the support files.

**A run without a scheduler now uses an existing bin directory.** `--no-schedule` never *creates* one — not even with `--create-bin` — but if the simulation already has one, the run executes the copy in it rather than the build directory's. So running a segment by hand inside a simulation runs the same binary its scheduled jobs do.

**An external submit script template is not frozen with the simulation.** A chain scheduled with `--submit-script-template` pointing outside the installation records that path in `SchedulerContext.yaml` and renders from it on every resubmission, so edits to it reach future segments and deleting it breaks them (`TemplateNotFound`). That follows from the settled design — a custom template is used where it is, not copied — but it is the one path on which a queued chain still depends on a file outside the simulation. It is documented rather than changed here; see the measurement under Testing.

**Builds configured with `BUILD_SHARED_LIBS=ON` can no longer schedule runs** unless you pass `--no-create-bin`. Creating a bin directory deliberately fails when the build directory is configured that way, because its executables load the SpECTRE libraries from there and a copy breaks as soon as the build directory changes (settled open point 6(a) of #7447). Four of the repo's own environment scripts configure exactly that: `support/Environments/urania.sh:53`, `support/Environments/viper.sh:55`, `support/Environments/ocean2.sh:80` and `support/Environments/ocean2_orca1.sh:63`. On Urania, Viper, Ocean2 and Ocean2_orca1 a standard build therefore hits the guard. See "Further comments" for why that is not resolved here.

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

The guard reads the build's own `CMakeCache.txt` and refuses when `BUILD_SHARED_LIBS` is set to a value CMake reads as true (`ON`, `TRUE`, `YES`, `1`, case-insensitively). A directory with no cache was never configured, and a cache without the entry says nothing, so both pass. Nothing new is plumbed through to the scheduler: the answer is already written down in the directory being copied.

It asks about the configuration rather than the files produced, because file names are not portable — a shared library is `.so` on Linux and `.dylib` on macOS — while the cache entry reads the same everywhere. It is also coarser than reading each executable's `NEEDED` entries, and deliberately so: it needs no `ldd` subprocess, and the condition it detects is the one that matters.

#### Other user-visible effects

- **New directories** `bin/` and `support/` in the pipeline or segments directory. The job's log header prints the bin directory.
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
ctest -R "support\.(BinDirectory|Python\.(Schedule|Main|RunNext)|DirectoryStructure|Machines)" --output-on-failure
# 100% tests passed, 0 tests failed out of 6
```

The bin-directory cases live in a new `tests/support/Python/Test_BinDirectory.py` (11 cases); `Test_Schedule.py` keeps the scheduling, resubmission and CLI cases (5). `test_bin_directory` covers:

- the contents of the bin directory, that the support directory next to it holds exactly `SUPPORT_FILES` and none of the files CMake generates in the build directory's, and that the executable is no longer in the segments directory;
- that neither the rendered `Submit.sh` nor `SchedulerContext.yaml` contains any path under the build directory's `bin` or `lib`, or under the source tree — the property this issue is about;
- that scheduling again reuses the bin directory instead of copying over it;
- that `<bin_dir>/spectre resubmit` works as a subprocess **with `PATH` and `PYTHONPATH` scrubbed**, so the build directory is unreachable through the environment, and writes bin-directory paths into the next segment. This exercises the relocatable wrapper and proves the copied package stands alone.

The others:

- `test_no_bin_directory_without_scheduler` — a directly executed run copies nothing, even with `create_bin=True`;
- `test_relocatable_build_guard` — the guard raises for every spelling of true (`ON`, `on`, `TRUE`, `Yes`, `1`) with the build directory in the message, and passes for the false ones, for a cache that doesn't mention the option, and for a directory with no cache at all;
- `test_bin_directory_search_stays_in_the_simulation` — a decoy `bin/` above a non-conforming directory is not picked up, and the same tree does find its own simulation's bin directory, including when that directory is a build directory's;
- `test_scheduling_into_a_build_directorys_bin` — scheduling a run tree that sits in a build directory uses the build's `bin`: nothing at all is copied, because the executable, the CLI and the support files are already in place, the build directory's `support/` is left untouched, and nothing fails;
- `test_installation_layout` — `this()` finds the build directory's `bin`, and the derived `python_dir` and `support_dir` are the same expressions anywhere else;
- `test_create` — copies a stand-in installation into a simulation with an explicit `source`, so it needs no mocking and no real package copy: the executable, CLI and Python directory land, the support files land by name with none of CMake's, duplicate executables are copied once, `executable()` reads what is there, and `add()` keeps a file already there rather than replacing it with a rebuilt one;
- `test_support_files_are_found_relative_to_the_installation` — `Machines.this_machine` defaults to the running installation's `Machine.yaml`, and the simulation's support directory holds exactly `SUPPORT_FILES`, intact and parsing;
- `test_custom_submit_script_template` — a template of your own is rendered where it is, is not copied into the simulation, and still resolves the installed base template it extends;
- `test_bin_directory_shared_by_the_simulation` — the steps of a pipeline and a branch nested inside the simulation all share the one bin directory: later ones reuse it instead of re-copying, and their submit scripts point there. It also covers an explicit opt-out surviving the handoff, and it caught a real bug: a later step passing its own submit script template raised `OSError: File already exists`, because only the executables had the "already there wins" rule. Copying into the bin directory now goes through one helper that applies it to the templates too;
- `test_cli_finds_the_package_next_to_itself` — running the wrapper in the build tree and running a copy of it elsewhere both put the package next to the script first on `sys.path`, with the configured entries following unchanged;

`test_schedule` now covers the `--no-create-bin` layout, which is also the layout of runs scheduled before this change, and ends by resubmitting a simulated pre-change `SchedulerContext.yaml` (no `create_bin`, obsolete `copy_executable` key, executable in the segments directory) to check that old runs keep working.

The Bbh pipeline tests pass `--no-create-bin` so they keep testing pipeline wiring without copying the Python package on every invocation; their expected `Next` blocks were updated for the renamed option. **These tests were not executed** — the pybindings they need don't link in the environment this was developed in, for reasons unrelated to this change (below). Their expectations were instead checked by rendering the pipeline templates directly and confirming that `create_bin` reaches the `Next` block as a boolean. **Please make sure CI runs `support.Pipelines.Bbh.*`.**

A test build usually selects no machine, so the unit tests point the running installation's `support_dir` at a stand-in directory. The production path was therefore also checked by hand with a real machine configured (`cmake -D MACHINE=CaltechHpc`): the default submit script template and `Machine.yaml` resolve to `<build_dir>/support/`, scheduling produces a byte-identical copy at `<simulation_dir>/support/` holding exactly those three files and none of CMake's, and `SchedulerContext.yaml` records the simulation's copy as the template.

Build-directory independence was checked in the same configuration, outside the test suite and outside the build directory: schedule a run, move the build directory away entirely, and resubmit through the bin directory's CLI.

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
