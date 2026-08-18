# Survey: simplify setup on a known machine

Surveyed revisions: SpEC sxs-collaboration/spec@653caf4736, SpECTRE sxs-collaboration/spectre@4d43624d64. All `file:line` below refer to those revisions.

## Verdict

**Not at parity, and the gap is structural rather than cosmetic.** SpEC has exactly one artifact per machine that everything reads — `MakefileRules/Machines/<Machine>.env`, selected once by `./configure` and thereafter symlinked as `this_machine.env`. The build reads it (`MakefileRules/RulesForEnv:27-29`), every submit script sources it (`Support/Python/MakeSubmit.py:166`), every segment continuation re-sources it (`Support/Perl/SpEC.pm:136-155`), and 45 of 50 of those files run `module purge` — 41 of them in their first three lines. A new user on a known machine runs `./configure && make parallel` and never touches their login shell.

SpECTRE splits the same information across **three or four files per machine with three different naming conventions**, requires the user to source one of them by hand in every shell, has no auto-detection, and loads no environment at all inside the batch job on 7 of its 10 registered machines. The concrete consequences, all verified below: `module purge` runs in 2 of 14 environment scripts and never inside a job; one machine (`Ocean`) is broken and no test catches it; three more have submit templates that cannot be selected; two environment scripts tell the user to edit `~/.bashrc`; and `CMakePresets.json` — which the issue asks about and which `AGENTS.md:8` already tells agents to use — does not exist in the repository (#7320 adds it and is open).

**This issue now also owns module version pinning**, per the settlement on #7447 (its open point 5, resolved to (a): the bin directory copies the environment script to `bin/env.sh`, and version pinning was deferred here).

**One item is time-critical and cross-cutting:** four machines set `BUILD_SHARED_LIBS=ON` in their `spectre_run_cmake` — `support/Environments/urania.sh:53`, `viper.sh:55`, `ocean2.sh:80`, `ocean2_orca1.sh:63`. #7447's settled open point 6 makes a build-directory shared-library dependency a **hard error at snapshot time**. Once #7447 lands, `spectre schedule` fails by default on Urania and Viper. See open point 8.

## What SpEC does

**Configuration is one command, and it auto-detects the machine.** `./configure` calls `Machines::GetHost()`, matches the hostname against a per-machine `RegExp` in the machine database (`Support/Perl/Machines.pm:1234-1259`, ~40 entries starting at `Machines.pm:168`), looks the host up in a name→file table (`configure:38-149`), and symlinks `MakefileRules/this_machine.def` and `this_machine.env` to the chosen machine files (`configure:189-216`). An unknown host is a hard error naming the hostname (`configure:167-173`). The only manual alternative is documented in three lines (`MakefileRules/README:1-3`).

**The build sources the environment itself; the user's shell is irrelevant.** `MakefileRules/RulesForEnv:27-29` runs `MakefileRules/env-diff` under `env -i` (a *cleared* environment keeping only `HOME`, `LOGNAME`, `USER`), which sources `/etc/profile`, runs `module purge`, sources `this_machine.env`, diffs the environment before and after, and writes the new variables as `export` lines into `environment.mk`, which the makefile includes (`RulesForEnv:16-18`). Consequences worth copying:

- **The user never loads modules to build.** `make` reconstructs the environment from the file.
- **The environment file is not allowed to print anything.** `env-diff:37-44` treats any output from sourcing it as a fatal error, with a two-machine exception list. A stray `module load` warning fails the build instead of silently changing it.
- **`module purge` has one documented exception mechanism**, a hostname test at `MakefileRules/env-diff:27-29` (BlueWaters and Trillian).
- **`LD_LIBRARY_PATH` from the modules is converted into `-Wl,-rpath` link flags** (`RulesForEnv:22-23`), so the built binary does not depend on the runtime environment reproducing the library search path.
- **A machine can declare the environment file mandatory.** `REQUIRE_ENVFILE = yes` in the machine `.def` turns a missing file into a hard error (`RulesForEnv:46-50`); 44 of 59 machine definitions set it, e.g. `MakefileRules/Machines/Urania.def:16`.

**Machine definitions compose.** `MakefileRules/Machines/Standards/` holds five shared fragments (`compilers_openmpi.def`, `flags_gcc.def`, …) included by the machine files — 31 defs include `compilers_openmpi.def`, 25 include `flags_gcc.def`. Retired machines move to `MakefileRules/Machines/Unsupported/` (7 entries), not out of the tree.

**`module purge` is the norm, and versions are pinned.** 45 of 50 `.env` files contain `module purge`, 41 of them within the first three lines (`MakefileRules/Machines/Urania.env:1`, `CaltechHpc_gccSXS.env:1`). The five without it are `Anvil_gcc.env`, `Edison_gcc.env`, `Juwels_gcc.env`, `Niagara_intel.env` and `WFUBinary.env`. Counting arguments to `module load`/`module add` across all 50 files: 345 of 426 carry an explicit version (`Urania.env:3` is representative: `intel/21.7.1 impi/2021.7 mkl/2022.1 hdf5-mpi/1.12.2 …`).

**Nothing asks the user to edit `~/.bashrc`.** The only mention says the opposite: `MakefileRules/Machines/Anvil_gcc.env:38-41` notes that whatever the user puts in `.bashrc`, sourcing the env file in a subshell overrides it.

**Every job entry point re-establishes the environment.** The generated submit header is `. bin/this_machine.env` followed by `export PATH=$(pwd -P)/bin:$PATH` (`Support/Python/MakeSubmit.py:162-168`) — unconditional, on every machine. `StartJob.sh` does the same (`Support/Perl/PrepareEv.pl:124`, `Support/Perl/PrepareID.pl:275`), and segment continuation re-sources it through `SpEC::LoadEnvFromBin` (`Support/Perl/SpEC.pm:136-155`, called at `Support/Perl/MakeNextSegment.pl:126`), which runs the file in a subshell and captures the resulting environment (`Support/Perl/Utils.pm:45-67`). Because the file begins with `module purge`, the job environment does not depend on the submitting shell.

**Containers are a machine property, not a separate workflow.** Setting `SPEC_USE_SINGULARITY_CONTAINER` in the env file makes `make` re-invoke itself inside the container (`MakefileRules/Rules:20-52`, `Support/bin/GetSingularityCmd`).

**Tested.** `Support/Tests/TestMachines/`, and `Support/Tests/TestPerlSpEC/TestPerlSpEC.pl:40-44` exercises `LoadEnvFromBin` against a fixture bin directory.

## What SpECTRE does today

**Three or four artifacts per machine, three naming conventions, no registry.**

| Artifact | Path | Naming | Count |
|---|---|---|---|
| Module environment + `spectre_run_cmake` | `support/Environments/<name>.sh` | lower_snake, sometimes with a compiler suffix (`caltech_hpc_gcc.sh`, `urania.sh`) | 14 |
| Scheduler metadata | `support/Machines/<Machine>.yaml` | CamelCase, must equal `MACHINE` | 10 |
| Submit template | `support/SubmitScripts/<Machine>.sh` | CamelCase, must equal `MACHINE` | 13 + base |
| CMake settings | — | — | 0 (proposed by #7320) |

The environment-script name does not follow from `MACHINE`, so anything that needs to find a machine's environment script from `MACHINE` needs a hard-coded mapping table. That is exactly what wiring #7447's `bin/env.sh` requires today.

**`MACHINE` is an unvalidated CMake option, used in one place.** `support/Python/CMakeLists.txt:8-9` declares it; lines 24-39 `configure_file` the YAML and the submit template into the Python package. The whole file returns early when `ENABLE_PYTHON` is off (`support/Python/CMakeLists.txt:4-6`), so `MACHINE` is silently ignored in that case. There is no auto-detection anywhere — the value comes from the hand-written `-D MACHINE=…` line in each `spectre_run_cmake`.

**One machine is broken.** `support/Environments/ocean_gcc.sh:95` passes `-D MACHINE=Ocean`, but neither `support/Machines/Ocean.yaml` nor `support/SubmitScripts/Ocean.sh` exists, so `configure_file` fails and CMake configuration aborts. The documentation still points at it (`docs/Installation/InstallationOnClusters.md:97-100`).

**Three machines have submit templates that can never be selected.** `anvil_gcc.sh`, `expanse_gcc.sh` and `frontera_gcc.sh` set no `MACHINE`, so `support/SubmitScripts/Anvil.sh`, `Expanse.sh` and `Frontera.sh` are never configured into the package; they are hand-edited scripts with `# Replace these paths with …` (`support/SubmitScripts/Expanse.sh:35-46`). The Anvil and Expanse environment scripts have not been touched since 2022-12-05.

**No environment is loaded inside the job on 7 of 10 registered machines.** Only `Urania.sh:25-30`, `Deucalion.sh:29-34` and `Viper.sh:25-29` source an environment script and call `spectre_load_modules`. `CaltechHpc.sh`, `Mbot.sh`, `Ocean2.sh`, `Ocean2_orca1.sh`, `Oscar.sh`, `Perlmutter.sh` and `Sonic.sh` contain no `module` command at all beyond the inherited `module list` (`support/SubmitScripts/SubmitTemplateBase.sh:44-46`). No template sets `#SBATCH --export`, so those jobs run under Slurm's default `--export=ALL` and inherit the submitting shell — the exact "works for me" mode the issue's third bullet names. (Slurm's default is documented behaviour, not measured here.)

**Where the three that do load an environment get it from is itself the #7447 defect:** `Urania.sh:28` and `Deucalion.sh:32` bake `@CMAKE_SOURCE_DIR@/support/Environments/<machine>.sh` in at configure time, and `Viper.sh:26` uses `${SPECTRE_HOME}`, which nothing in the template sets.

**`module purge` appears twice in the repository**, both inside `spectre_load_modules`: `support/Environments/ocean_gcc.sh:50` (the broken machine) and `support/Environments/oscar.sh:15,29`. Neither runs in a batch job, because neither machine's submit script calls `spectre_load_modules`. There is one documented counter-example: `docs/Installation/InstallationOnClusters.md:48-50` says of Anvil, "Avoid running `module purge` because this also removes various default modules that are necessary for proper operation. Instead, use `module restore`."

**Module versions are pinned about two-thirds of the time.** Counting arguments to `module load` across `support/Environments/*.sh`: 111 of 172 carry an explicit `name/version`. The unpinned ones cluster in the three unregistered machines (anvil 12, expanse 15, frontera 13) plus `mbot.sh:13` (`spectre-deps`), `sonic.sh:12-13` (`sxs`, `spectre-env`) and `perlmutter.sh:8,10`. Two caveats: `ocean_gcc.sh:62-78` uses Spack hash-suffixed module names, which are pinned *more* tightly than a version and must not be flagged by any lint; and `urania.sh:7` loads `gcc/11`, a floating major-version alias.

**`~/.bashrc` edits are suggested in six places.** Two are printed by the environment scripts themselves: `caltech_hpc_gcc.sh:8-11` and `oscar.sh:8-11` both echo "Place the following line in your `~/.bashrc` so you don't have to run `spectre_setup_modules` every time you log in". Both are already redundant — the same `module use` is repeated inside `spectre_load_modules` (`caltech_hpc_gcc.sh:15-16`, `oscar.sh:16-17`). Three more are printed by the dependency installers (`support/Environments/setup/anvil_gcc.sh:240`, `expanse_gcc.sh:311`, `frontera_gcc.sh:216`), which ask the user to persist a `module use $SPECTRE_DEPS/modules`. The sixth is `docs/Installation/InstallationOnClusters.md:130-131` for Sonic. A seventh, `docs/Tutorials/Cli.md:34-35` (shell completion), is a genuine convenience and should stay.

**What a new user on a known machine does today**, walking `docs/Installation/InstallationOnClusters.md:19-39` end to end: export `SPECTRE_HOME`; clone; `mkdir build`; source `support/Environments/<SYSTEM>_gcc.sh` — a name the user must guess, since only 5 of 14 scripts carry the `_gcc` suffix the instructions assume, and the Sonic section asks for `Environments/Sonic.sh` while the file is `sonic.sh` (`InstallationOnClusters.md:132`); possibly build dependencies; `module use`; `spectre_run_cmake`; `make`. Every step after the clone must be repeated in every new shell, and 7 of the 14 machines — including Urania and Viper, the two the parity campaign runs on — have no section in that document at all.

**Presets do not exist yet.** No `CMakePresets.json` is tracked at the surveyed revision; the only mention of presets in the repository is `AGENTS.md:8`, which instructs agents to prefer presets that are not there. `.gitignore:13` already ignores `CMakeUserPresets.json`. #7320 adds the missing file and is open.

**Negative results.** No container is used on clusters: `apptainer`, `singularity` and `container` appear nowhere in `docs/Installation/InstallationOnClusters.md`, `support/Environments/` or `support/SubmitScripts/` (SpECTRE's containers, `docs/Installation/Installation.md:22-92,306-338`, target releases and laptop development). No test validates the real machine files — `tests/support/Python/Test_Machines.py:17-32` writes a synthetic YAML and never reads `support/Machines/`. No lint applies to environment scripts beyond a long-line exclusion (`tools/FileTestDefs.sh:302`). There is no equivalent of `env-diff`, of `REQUIRE_ENVFILE`, of `Standards/`, or of `Unsupported/`.

**One thing SpECTRE already does better:** CMake emits a `RUNPATH` for dependencies found outside the standard search path — measured with `readelf -d` on a release-build executable in this environment, which carries a `RUNPATH` entry for its HDF5 directory. That is SpEC's `MODULE_RPATHS` (`RulesForEnv:22-23`) obtained for free, and it is why an unloaded module environment degrades gracefully rather than instantly. Formaline's provenance record (`printenv` and `BuildInfo.txt` into every executable and H5 file, `docs/Installation/BuildSystem.md:540-556`) is also stronger than SpEC's `bin/env.log` / `bin/module.log` (`Support/Perl/SpEC.pm:130-131`) — but, as #7447 already found, neither code *acts* on the record.

## Prior art

| Ref | State | Action |
|---|---|---|
| #7320 "Use CMake presets" | open PR | **Read first; this issue builds on it.** Adds `CMakePresets.json` including `support/Environments/$penv{SPECTRE_MACHINE}Presets.json`, a shared `Presets.json` with `debug` / `release` / `release-debug`, an Mbot example, and rewrites `spectre_run_cmake` to `cmake -S … -B . --preset release-debug`. `spectre_load_modules` gains `export SPECTRE_MACHINE=Mbot`. Its own description names the next step as out of scope: "In the future we can expand the use of presets further to load modules etc, though that's a larger change and breaks the existing interface". That larger change is this issue. Recommend merging #7320 as-is first and rebasing this work on it. |
| #6471 "Add CMake Presets for mbot" | open draft PR | **Close as superseded by #7320, carry the idea forward.** Predates #7320 (2025-02) and takes the more radical route: a `MbotCommon.cmake` toolchain file listing absolute `CMAKE_PREFIX_PATH` entries, so no modules are used at all. @nilsdeppe in #7320: "a possibility is to completely eliminate module use except for a CMake module. Everything then goes through cmake". @nilsvu: "a useful path forward to migrate more/all of our current shell scripts into cmake presets". This is option (c) of open point 3. |
| #7447 "Bin directories" | open, in progress | **Compose.** Its settled open point 5 copies the machine environment script to `bin/env.sh` and sources it from the submit template; module version pinning was explicitly deferred to this issue. Its settled open point 6 (hard error on build-directory shared libraries) collides with the four machines that set `BUILD_SHARED_LIBS=ON` — open point 8 below. |
| #7444 | closed as duplicate | The `module purge` topic; already folded into this issue's third bullet. |
| #442 "Supercomputer support for compilers and python" | open | **Close.** Opened 2017 to raise the minimum GCC version; the requirement was met long ago. What remains is a hand-maintained table of per-cluster compiler/CMake/Python versions, last meaningfully edited in 2022 and now wrong (it lists Comet, retired 2021). A machine registry with a completeness test replaces it. |
| #3699 "Cluster environments should treat dependencies as system-installed" | open | **Link.** Asks for `CPATH` → `C_INCLUDE_PATH` / `CPLUS_INCLUDE_PATH` in the environment files to silence dependency warnings; @wthrowe listed the affected dependencies in-thread. It is a per-machine edit to the same files this issue restructures — fold it into the migration rather than doing it twice. |
| #6331 "Various problems with regards to Installation of spECTRE" | open | **Link, do not merge.** An outside user's account of failing to install via Spack and Docker. Different audience (no known machine, no modules), same symptom class. |
| #7441 "Connect to BFI" | open | **Link.** The issue body calls this topic "sort of BFI-related, but not part of BFI"; keeping the boundary visible on both cards is enough. |
| #5100, #6351, #5271 | open | Symptoms of environment drift (OpenBLAS thread counts, pybindings not found after CMake, `-lfftw3` not found on Ocean). Context only; each may close on its own once machines are pinned. |
| #2201, #2470, #2947, #3324, #3712, #3968, #5725, #6311, #6710, #5750 | merged | The per-machine environment-file churn this issue exists to reduce. One line, no action. |

Searched issues and PRs repo-wide (open and closed) for `module purge`, `spectre_setup_modules`, CMake presets, environment files, bashrc, spack, modules, submit scripts, and each machine name. Nothing else actionable; in particular there is no existing issue for the broken `MACHINE=Ocean`, for the missing job-start environment on 7 machines, or for module version pinning.

## Proposed design

**One directory per machine is the artifact.** Replace the three parallel file sets with `support/Machines/<Machine>/`, fixed file names, `<Machine>` equal to the `MACHINE` value:

```
support/Machines/<Machine>/
  Machine.yaml     # scheduler metadata (today support/Machines/<Machine>.yaml), plus new keys below
  Env.sh           # spectre_load_modules / spectre_unload_modules / spectre_setup_modules
  Presets.json     # CMake configure presets (#7320's <Machine>Presets.json)
  Submit.sh        # submit template (today support/SubmitScripts/<Machine>.sh)
  Local.sh         # optional, git-ignored: site-local paths (see open point 6)
```

This is what makes everything else cheap: "is this machine complete?" becomes a directory listing, "where is this machine's environment script?" becomes `support/Machines/${MACHINE}/Env.sh` with no mapping table, and adding a machine becomes copying one directory.

**Two new `Machine.yaml` keys.**

- `HostnameRegex:` — a regular expression matching the machine's login and compute node hostnames, SpEC's mechanism (`Support/Perl/Machines.pm:1243-1256`). This is what enables auto-detection.
- `ModuleReset:` — `purge` (default), `restore`, or `none`. Anvil is the documented reason this cannot be a global constant (`docs/Installation/InstallationOnClusters.md:48-50`).

**One entry script the user sources, `support/LoadEnv.sh`**, machine-independent and the same on every machine:

1. resolve `SPECTRE_HOME` from its own location, so it works from any directory and any worktree;
2. set `SPECTRE_MACHINE` by matching the hostname against every `HostnameRegex` (error naming the hostname if none or more than one matches, as `configure:167-173` and `Machines.pm:1252-1254` do), unless `SPECTRE_MACHINE` is already set — the escape hatch for a machine whose hostname is ambiguous;
3. run `module purge` / `module restore` / nothing per `ModuleReset`;
4. source `support/Machines/$SPECTRE_MACHINE/Env.sh` and call `spectre_load_modules`;
5. source `support/Machines/$SPECTRE_MACHINE/Local.sh` if it exists.

The whole known-machine story then reads:

```sh
. ./support/LoadEnv.sh
cmake --preset release-debug
cmake --build build-release-debug -j4 -t EvolveGhBinaryBlackHole
```

Implementation detail for whoever writes it: resolving a *sourced* script's own path is shell-specific (`${BASH_SOURCE[0]}` in bash, `${(%):-%x}` in zsh, unavailable in POSIX `sh`). The existing scripts sidestep this by requiring the user to set `SPECTRE_HOME` (e.g. `support/Environments/urania.sh:41-44`); keeping that as the fallback when auto-resolution fails is fine.

**No-bashrc principle, stated and enforced.** The repository never requires an edit to a login file; anything persistent lives in the machine directory or in `Local.sh`. Concretely: delete the two `echo "Place the following line in your '~/.bashrc'"` blocks (`caltech_hpc_gcc.sh:8-11`, `oscar.sh:8-11`) — the `module use` they advertise is already inside `spectre_load_modules` at both places, so nothing is lost. Change the three installer tails (`support/Environments/setup/*.sh`) to write `Local.sh` instead of printing an instruction. Keep the shell-completion suggestion (`docs/Tutorials/Cli.md:34-35`).

**Module version pinning.** Rule: every `module load` in `Env.sh` names an explicit version, either `name/version` or a Spack hash-suffixed name. Enforced by a new check in `tools/FileTestDefs.sh`, the repository's existing staged-file checker, scoped to `support/Machines/*/Env.sh`. Meta-modules are pinned by dated version, as CaltechHpc already does (`caltech_hpc_gcc.sh:17`, `spectre-deps/2025-09`); `mbot.sh:13`, `sonic.sh:12-13`, `perlmutter.sh:8,10` and `urania.sh:7` are the ones that need owner input. This is the piece #7447's settlement deferred here.

**Job-start environment, unconditional.** `support/SubmitScripts/SubmitTemplateBase.sh` gets one `list_modules` block that applies to every machine: the `ModuleReset` line, then the environment (`bin/env.sh` when a bin directory exists — #7447's hook — otherwise `support/Machines/${MACHINE}/Env.sh` at its configure-time path), then `spectre_load_modules`, then `module list`. The three machine templates that currently do this themselves (`Urania.sh:25-30`, `Deucalion.sh:29-34`, `Viper.sh:25-29`) drop their overrides and inherit; the other seven gain a defined environment they do not have today. This is `Support/Python/MakeSubmit.py:162-168` — one unconditional line in the header — and it is what makes a job's environment independent of the submitting shell.

**How this feeds #7447.** #7447 copies the machine environment script into `bin/env.sh` and sources it from the submit template. With the machine directory, the CMake side becomes one unconditional `configure_file` of `support/Machines/${MACHINE}/Env.sh` — no machine→filename table to keep in sync. Recommend `bin/env.sh` be *generated* rather than copied verbatim: a small wrapper that performs the `ModuleReset` step and then sources the machine's `Env.sh` next to it, so the snapshot carries the reset decision and stays self-describing after the source tree has moved on. The two changes are independent and can land in either order; if #7447 lands first, this issue collapses its mapping table.

**Registered / unregistered / retired.** Every directory under `support/Machines/` is complete or the machine moves to `support/Machines/Unsupported/<Machine>/`, SpEC's model (`MakefileRules/Machines/Unsupported/`, 7 entries) — kept in the tree, excluded from the completeness test and from the documentation. Anvil, Expanse and Frontera (no `MACHINE`, environment scripts untouched since 2022 for two of them) and Ocean (broken) are the candidates; see open point 7.

**Migration.** Mechanical and reviewable per machine: `git mv support/Machines/<M>.yaml support/Machines/<M>/Machine.yaml`, `git mv support/SubmitScripts/<M>.sh support/Machines/<M>/Submit.sh`, `git mv support/Environments/<name>.sh support/Machines/<M>/Env.sh`, add `Presets.json`. `MACHINE=<Name>` keeps its spelling, so existing build directories, `SchedulerContext.yaml` files and running simulations are unaffected. `support/Environments/` is left holding only `setup/` (the dependency installers) or is removed entirely. Documentation: `docs/Installation/InstallationOnClusters.md` shrinks to the generic recipe plus a per-machine table generated from `Machine.yaml`'s `Description`, which is where that text already belongs (`support/Python/Machines.py:40-43` documents the field as exactly this).

**Testing / acceptance.**

- Extend `tests/support/Python/Test_Machines.py` to walk `support/Machines/*/`: assert the four files exist, `Machine.yaml` parses into a `Machine` whose `Name` equals the directory name, `HostnameRegex` compiles, `ModuleReset` is one of the three values, and no two machines' regexes match the same sample hostname. This is the test that would have caught `MACHINE=Ocean`.
- New `tools/FileTestDefs.sh` check: unversioned `module load` in `support/Machines/*/Env.sh`.
- Presets: `cmake --list-presets` for each machine's `Presets.json` — schema and inheritance only, no configure, so it runs on CI without the machines.
- Rendered-submit assertion in `tests/support/Python/Test_Schedule.py`: the rendered `Submit.sh` contains the module-reset line and sources an environment script, for every machine template.
- Acceptance on real hardware: on Urania and on Viper, from a login shell with no modules loaded, `. ./support/LoadEnv.sh && cmake --preset release-debug && cmake --build … -t EvolveGhBinaryBlackHole` succeeds; a job scheduled from that shell prints the same `module list` in its output as the build did; and the same job scheduled from a shell with a *different* module set prints the same list again. That third check is the property the issue's third bullet asks for and nothing tests today.

## Open points to settle

- [ ] **1. What artifact encodes a known machine** — (a) one directory per machine holding `Machine.yaml` + `Env.sh` + `Presets.json` + `Submit.sh`; (b) keep the three parallel directories and only fix the naming so the environment script's name follows from `MACHINE`; (c) keep everything as is and add a registry file listing each machine's four paths. **Recommend (a).** SpEC's single `this_machine.env` is the reason its setup is one command; (b) fixes the mapping table but still leaves "add a machine" as three edits in three directories with nothing checking they agree, which is how `MACHINE=Ocean` (`support/Environments/ocean_gcc.sh:95`) came to reference two files that do not exist.
- [ ] **2. Machine identification** — (a) hostname auto-detection from a `HostnameRegex` in `Machine.yaml`, with `SPECTRE_MACHINE` as an override; (b) `SPECTRE_MACHINE` only, set by the environment script as #7320 does; (c) keep the hand-written `-D MACHINE=` in each `spectre_run_cmake`. **Recommend (a).** It is what lets one command work on every machine without the user knowing the machine's spelling, it is SpEC's mechanism (`Support/Perl/Machines.pm:1234-1259`, `configure:151`), and the override covers ambiguous hostnames — a case SpEC hit and solved by giving Urania's test partition its own entry (`Support/Perl/Machines.pm:598,612`). Note that regexes need maintenance when a site renames nodes; the failure mode is a clear error, not a wrong build.
- [ ] **3. How much moves into CMake** — (a) presets carry the CMake variables, modules stay in `Env.sh` (this proposal, and #7320's direction); (b) additionally a per-machine CMake toolchain file with absolute `CMAKE_PREFIX_PATH` entries, so no modules are needed to configure (#6471's `MbotCommon.cmake`); (c) eliminate modules entirely, everything through CMake. **Recommend (a) now, (b) as a per-machine option.** (c) does not solve the runtime environment: a batch job still needs `LD_LIBRARY_PATH`, `PYTHONPATH` and an interpreter, and modules are how the sites deliver them. (b) is genuinely attractive for sites with stable install paths and can be adopted machine by machine without changing the layout.
- [ ] **4. `module purge` default** — (a) purge by default, with `ModuleReset: restore|none` per machine; (b) never purge, document that users should start from a clean shell; (c) purge only inside batch jobs, not interactively. **Recommend (a).** SpEC does it in 45 of 50 machines and additionally before its build-time environment diff (`MakefileRules/env-diff:27-29`), with an explicit exception list — the same shape. Anvil is the known exception (`docs/Installation/InstallationOnClusters.md:48-50`). (c) is tempting because interactive purge is the more disruptive half, but it splits the build environment from the run environment, which is the thing this issue is trying to stop.
- [ ] **5. Does the job-start environment change apply to all machines at once?** Seven of ten registered machines currently inherit the submitting shell (`CaltechHpc.sh`, `Mbot.sh`, `Ocean2.sh`, `Ocean2_orca1.sh`, `Oscar.sh`, `Perlmutter.sh`, `Sonic.sh` load nothing). Giving them a defined environment is the point of the issue, but it changes behaviour for anyone whose current runs depend on their shell. Options: (a) switch all machines in one PR; (b) switch machine by machine as each owner confirms; (c) add the block behind an opt-in `Machine.yaml` key and flip machines over time. **Recommend (b)**, with Urania, Viper and Deucalion first since they already load an environment and are the campaign's machines. Whoever owns each remaining machine should confirm before its flip — this survey cannot establish what those sites' default module sets contain.
- [ ] **6. Where site-local settings live now that `~/.bashrc` is out** — the dependency installers currently ask the user to persist `module use $SPECTRE_DEPS/modules` (`support/Environments/setup/anvil_gcc.sh:240` and the two beside it). Options: (a) a git-ignored `support/Machines/<Machine>/Local.sh` sourced by `LoadEnv.sh` if present; (b) `~/.spectre/<machine>.sh`, outside the repository so it survives worktrees and re-clones; (c) `CMakeUserPresets.json` only, which covers build variables but not module paths. **Recommend (b)** — the value being stored (`$SPECTRE_DEPS`) is per user and per machine, not per checkout, and every worktree needing its own copy is a papercut the campaign already lives with. (a) is simpler to implement and discover; either is acceptable, (c) alone is not sufficient.
- [ ] **7. Retire or repair the four incomplete machines** — Anvil, Expanse and Frontera have submit templates and environment scripts but set no `MACHINE`, so their templates are unreachable; Anvil's and Expanse's environment scripts were last changed 2022-12-05. `Ocean` sets `MACHINE=Ocean` with no `Machine.yaml` and no submit script, which fails CMake configuration outright. Options: (a) move all four to `Unsupported/`; (b) complete them; (c) leave them. **Recommend (a) for Anvil, Expanse and Frontera**; `Ocean` needs an owner's answer first — Ocean2 exists and is maintained (`support/Environments/ocean2.sh`, last changed 2025-10-22), so `support/Environments/ocean_gcc.sh` may simply be a leftover, but that is a guess and should be confirmed rather than assumed.
- [ ] **8. `BUILD_SHARED_LIBS=ON` on four machines vs #7447's snapshot guard** — `support/Environments/urania.sh:53`, `viper.sh:55`, `ocean2.sh:80` and `ocean2_orca1.sh:63` build shared libraries; #7447's settled open point 6 makes an executable that loads shared objects from the build directory a hard error at snapshot time, so `spectre schedule` will fail there by default once #7447 merges. Options: (a) drop `BUILD_SHARED_LIBS=ON` from the production presets on those machines, keeping it available as a developer preset; (b) keep it and have those machines schedule with `--no-create-bin`, forfeiting the snapshot; (c) weaken #7447's guard to a warning. **Recommend (a)**, and recommend splitting it out of this issue as its own small change so it can land before #7447 rather than after — it is four one-line edits, and the reason each machine turned the flag on (link time? memory during linking?) is not recorded anywhere in the repository and should be asked of the machines' owners.
- [ ] **9. Sequencing against #7320** — (a) merge #7320 as-is, then rebase this restructure on it (its preset files move into the machine directories); (b) fold #7320 into this issue and land one change; (c) do this restructure first and rebase #7320. **Recommend (a).** #7320 is reviewed, self-contained, and changes no existing behaviour; #6471 closes as superseded either way.

A follow-up comment settling these points makes this issue ready for implementation (→ Ready).

---

🤖 drafted by: [Claude Fable 5](https://claude.com/claude-code)
(and reviewed by me 🙋)
