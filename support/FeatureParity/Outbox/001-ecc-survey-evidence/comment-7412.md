# Survey: initial orbital parameters — SpEC's `ZeroEccParamsFromPN` vs SpECTRE vs the SimulationSupport plan

Feature-parity survey evidence. Revisions: SpEC @ `5f8f5375ca`, SpECTRE
`develop` @ `4d43624d64`, `sxs-collaboration/SimulationSupport` `main`
(2026-08-13). All `file:line` references are to these revisions.

**Framing.** The plan of record is to wire the new repository
`sxs-collaboration/SimulationSupport` into **both** SpEC and SpECTRE as the
provider of initial orbital parameters, with GPR-fitted initial guesses being
added there as an alternative to the PN guesses. PR #6224
("Use PostNewtonian.jl for initial orbital parameters") is recorded below as
context for what was previously attempted, not as the plan.

## What SpEC has

`Support/Python/ZeroEccParamsFromPN.py` (259 lines), exposed as the
executable `Support/bin/ZeroEccParamsFromPN` (registered in the bin
directory at `Support/Perl/SpEC.pm:104`).

Docstring (`ZeroEccParamsFromPN.py:3-13`):

> Determines low eccentricity initial parameters (D0, Omega0, adot0) from
> the 3.5 Post-Newtonian expressions for r vs. omega and rdot vs. r for
> circular orbits given some criterion to satisfy, i.e. desired initial
> separation, frequency, or number of orbits. The number of orbits is
> determined by integrating 3.5PN (T4) expressions.
> This will typically be a crude estimate for SpEC initial data parameters,
> but should be good enough to start the eccentricity reduction process.

### The two kernels

1. **`omegaAndAdot(r, q, chiA, chiB, rPrime0)`**
   (`ZeroEccParamsFromPN.py:27-63`) — closed-form 3.5PN `Ω(r)` and
   `adot = ṙ/r` for circular orbits. Terms `A1, A1p5, A2, A2p5, A3, A3p5`
   at `:45-51`; radial term `B1, B1p5, B2` at `:56-61`. Provenance is
   documented inline (`:40-44,54-55`): eq. 4.2 of arXiv:1212.5520v1 for
   `Ω(r)`, eq. 4.5 of gr-qc/9506022 for the 2PN spin-spin term, eq. 4.12
   of gr-qc/9506022 for `ṙ`. `rPrime0` is an explicit gauge choice —
   `:193`: *"rPrime0 is a gauge choice, try a couple different values and see
   what the difference is"* (main tries `[1., 10.]`).

2. **`nOrbitsAndTotalTime(q, chiA0, chiB0, omega0, cutoffFrequency=0.1)`**
   (`:66-158`) — integrates a 3.5PN T4 precessing ODE system (`dydt` at
   `:85-146`, with spin and `L_N` precession) via `scipy.integrate.odeint`
   from `omega0` to `cutoffFrequency = 0.1`, returning `(orbits, totalTime)`.
   The PN coefficients are annotated `:105`: *"Taken from Triton, should be
   updated. Could interface with GWFrames."*

### The CLI

`main()` (`:192-255`): required `--q`, `--chiA`, `--chiB`; then **exactly one
of** `--D0`, `--Omega0`, `--NOrbits`, `--tMerger` (mutually exclusive group,
`:203-212`). For `NOrbits`/`tMerger` it root-finds `omega0` with
`scipy.optimize.fmin` (`:228-242`), then calls `fromOmega0` (`:168-189`),
which root-finds `D0` for that `Ω0` and evaluates `adot0`.

### How it is (not) wired into SpEC's pipeline

**It is a standalone helper, not a pipeline stage.** `Params.input` leaves
the three parameters blank for the user —
`InputFiles/BbhID_AMR/Params.input:9-11`:

```perl
$Omega0 = ;
$adot0 = ;
$D0 = ;
```

Grep for callers across the whole checkout finds only
`Support/Python/GrHydro_ID_script_functions.py:57,92` (BHNS/NSNS ID scripts
shelling out to `./bin/ZeroEccParamsFromPN`) and the bin registration at
`Support/Perl/SpEC.pm:104`. **No BBH driver calls it.** The BBH workflow is:
human runs `ZeroEccParamsFromPN`, human pastes `Omega0`/`adot0`/`D0` into
`Params.input`, `PrepareID -reduce-ecc` (`Support/Perl/PrepareID.pl:113,192`)
sets `EccRedRun=1`, and eccentricity reduction takes it from there.

Once in `Params.input`, the values propagate to `ID_Params.perl` as
`$ID_Omega0`/`$ID_adot0` and seed the maps —
`InputFiles/Bbh/SpatialCoordMap.input:15-16` (`DtYaw = __Omega__`,
`DtExpansion = __aDot__`), filled at
`InputFiles/Bbh/DoMultipleRuns.input:1106-1107`.

## What SpECTRE has today

**SpECTRE already has a function named `initial_orbital_parameters`, and it
is already the wrapper the plan needs — it just calls SpEC for the physics.**

`support/Pipelines/EccentricityControl/InitialOrbitalParameters.py:15-198`:

```python
def initial_orbital_parameters(
    target_params: dict,
    separation: Optional[float] = None,
    orbital_angular_velocity: Optional[float] = None,
    radial_expansion_velocity: Optional[float] = None,
) -> Tuple[float, float, float]:
```

Reads `MassRatio`, `DimensionlessSpinA`, `DimensionlessSpinB`,
`Eccentricity`, `MeanAnomalyFraction`, `NumOrbits`, `TimeToMerger` from
`target_params` (`:64-70`). Same "exactly one of separation / Ω0 / NumOrbits
/ TimeToMerger" contract as SpEC (`:99-108`).

The SpEC dependency is two lines (`:110-115`):

```python
# Import functions from SpEC until we have ported them over. These functions
# call old Fortran code (LSODA) through scipy.integrate.odeint, which leads
# to lots of noise in stdout. When porting these functions, we should
# modernize them to use scipy.integrate.solve_ivp.
check_spec_import()
from ZeroEccParamsFromPN import nOrbitsAndTotalTime, omegaAndAdot
```

The rest is SpECTRE's own: Nelder-Mead root-finds replacing SpEC's `fmin`
(`:118-143` for `NumOrbits`/`TimeToMerger`, `:146-167` for separation), a
consistency check `np.isclose(..., rtol=1e-4)` (`:180-185`), and
`rPrime0=1.0` pinned with the comment *"Choice also made in SpEC"*
(`:154,175`).

Restriction (`:85-88`): `assert eccentricity == 0.0`.

Call site: `support/Pipelines/Bbh/InitialData.py:13-15` (import), `:298-310`
(invoked from `generate_id` when any of the three orbital parameters is
`None`).

### The dependency mechanism

`support/Python/CheckSpecImport.py:8-40` — tries `import Utils as spec_utils`
(`:18`), and on failure raises

> Importing from SpEC failed. Make sure you have pointed '-D SPEC_ROOT' to a
> SpEC installation when configuring the build with CMake.

with an optional commit-ancestry version check via
`git merge-base --is-ancestor` (`:25-40`).

The path is injected at build time — `cmake/SetupSpec.cmake:4-11`:

```cmake
find_package(SpEC)
...
if (SPEC_ROOT)
  set(PYTHONPATH "${SPEC_ROOT}/Support/Python:${SPEC_ROOT}/Support/DatDataManip:${PYTHONPATH}")
endif()
```

with `cmake/FindSpEC.cmake:10-13` reading `SPEC_ROOT` from the environment.

### Consequence: none of this is tested in CI

`tests/support/Pipelines/EccentricityControl/CMakeLists.txt:4-17` wraps
**both** ecc-control tests in `if (SpEC_FOUND)`; the BBH ecc-control test is
gated the same way (`tests/support/Pipelines/Bbh/CMakeLists.txt:28-35`).
Grep over `.github/workflows/` finds no `SPEC_ROOT` and no SpEC checkout, so
`SpEC_FOUND` is false in CI and **`initial_orbital_parameters` and the
eccentricity fit are never exercised there**.

This is the strongest technical argument for the SimulationSupport plan
independent of any physics improvement: a pip-installable dependency makes
these tests run.

## What the SimulationSupport plan requires

The wiring SpECTRE needs is narrow, because the wrapper already exists:

1. Replace `check_spec_import()` +
   `from ZeroEccParamsFromPN import nOrbitsAndTotalTime, omegaAndAdot`
   (`InitialOrbitalParameters.py:114-115`) with a SimulationSupport import.
2. Decide whether SpECTRE keeps its own wrapper
   (`initial_orbital_parameters`, root-finds and all) and calls
   SimulationSupport only for the PN kernels, **or** delegates the whole
   function to SimulationSupport's `initial_orbital_parameters` and becomes a
   thin adapter from `target_params` to that signature.
3. Add the dependency where SpECTRE declares Python requirements, and drop
   the `if (SpEC_FOUND)` gate on the tests so they run in CI.
4. The same swap in `EccentricityControlParams.py:92-99`
   (`from OmegaDotEccRemoval import ...`) is a *separate* change — that is
   the measurement side, tracked in #7416 and attempted in PR #6890.

Both open questions are answered by the survey below:
- Item 2 is settled — SimulationSupport's copy is byte-compatible with
  SpECTRE's, so **delete SpECTRE's copy and import** (exactly what PR #6890
  does). No adapter is needed.
- The GPR does **not** change the signature; it is an additive correction on
  top of the PN guess and is not wired into `initial_orbital_parameters` at
  all. So the wiring done now will not need redoing when the GPR lands —
  but somebody still has to do that wiring, and no PR proposes it.

## What SimulationSupport actually provides today

Surveyed at `sxs-collaboration/SimulationSupport` (public, MIT, created
2025-10-02, last push 2026-08-06, default branch `main`).

**The headline: SimulationSupport's `initial_orbital_parameters` is a
near-verbatim copy of SpECTRE's, and it carries SpEC's
`ZeroEccParamsFromPN.py` inside the package.** It does not use
PostNewtonian.jl.

- `src/SimulationSupport/EccentricityControl/InitialOrbitalParameters.py:14-19`
  — signature **identical** to SpECTRE's
  (`target_params, separation, orbital_angular_velocity,
  radial_expansion_velocity`). Differences from SpECTRE's copy: docstring
  converted to NumPy/Sphinx style, and `check_spec_import()` replaced by a
  relative import at `:112`:
  `from .ZeroEccParamsFromPN import nOrbitsAndTotalTime, omegaAndAdot`.
- `src/SimulationSupport/EccentricityControl/ZeroEccParamsFromPN.py` (273
  lines) — SpEC's file moved over almost unchanged (header lines 1-18 say so;
  marked `# fmt: off` / `# isort: skip_file`). Same 3.5PN content, same
  `odeint`/LSODA integration, same provenance citations at `:54-58`.
- Its test asserts the **SpEC** numbers
  (`tests/EccentricityControl/Test_InitialOrbitalParameters.py:20`:
  `[16.0, 0.014474280975952748, -4.117670632867514e-05]`) — byte-identical to
  what SpECTRE develop asserts today.

So adopting SimulationSupport for the initial orbital parameters is a **pure
refactor with no numerical change**: SpECTRE deletes its copy and imports the
identical function from a package instead of from a `SPEC_ROOT` PYTHONPATH
injection.

### The GPR work

Real, by Vittoria Tommasini, and **mostly already merged into `main`**
(5 merged PRs between 2026-05-07 and 2026-08-06; the repo's only other
contributor is infrastructure work).

- `src/SimulationSupport/gpr/core.py` (469 lines) — `GPRegressionModel`
  (`:20`) on **gpytorch/torch** (not sklearn): `ScaleKernel(RBFKernel)` +
  `ScaleKernel(MaternKernel(nu=2.5))`, `LinearMean`, with normalization
  baked in. `train_gpr_model` `:110`, `predict_with_gpr_model` `:226`,
  `run_gpr_pipeline` `:264`, `save_gpr_checkpoint` `:342`,
  `load_gpr_checkpoint` `:395`.
- `src/SimulationSupport/gpr/diagnostics.py` (252 lines) — leave-one-out
  cross-validation and residual plots.
- `.../EccentricityControl/Examples/GPRTutorial.ipynb` documents the physics:
  trained on `SXS:BBH:1419`–`1509` filtered to `reference_eccentricity <=
  1e-3` (89 points); 4 features (`initial_separation`,
  `initial_mass_ratio`, `initial_dimensionless_spin1_z`,
  `initial_dimensionless_spin2_z`).

**Critical API fact: the GPR is an additive correction on top of the PN
guess, not a replacement.** Two independent GPs are trained on *residuals*
`Y_omega = initial_orbital_frequency − pn_guess_omega` and
`Y_adot = initial_adot − pn_guess_adot`, with the baseline coming from
`ZeroEccParamsFromPN.omegaAndAdot` at `rPrime0 = 1`. Inference is
`omega_pn + delta_omega`, `adot_pn + delta_adot`, each with a 1σ
uncertainty.

**The GPR is not wired into `initial_orbital_parameters`.** That function's
signature is unchanged; nothing in `gpr/` imports it and it does not import
`gpr/`. No branch or PR proposes the wiring. Trained checkpoints
(`gpr_model_omega.pth`, `gpr_model_adot.pth`) are not committed — the
notebook trains them locally.

### Packaging — the real blocker for depending on it

- **Not on PyPI** (both `SimulationSupport` and `simulation-support` 404).
  No conda recipe.
- **No tags, no releases.** `version = "0.1.0"` (`pyproject.toml:8`) frozen
  since the initial commit.
- Dependencies (`pyproject.toml:12-21`, `requires-python >= 3.9`):
  `numpy, scipy, matplotlib, torch, gpytorch, pandas, sxs`. **`torch` and
  `gpytorch` are unconditional** even though only `gpr/` needs them, and
  `sxs` is a hard dependency that nothing on `main` imports outside the
  tutorial notebook. Adopting the package as-is drags all of that into every
  SpECTRE Python environment.
- The project's own documented workflow is a **pinned commit hash** —
  `docs/index.rst:28-31`: *"Once the pull request is merged, update the
  SimulationSupport version hash used by SpEC and update SpEC to import the
  file from SimulationSupport instead of the local copy"*.

## Prior art

### PR #6890 — "Ecc control: depend on SimulationSupport instead of SpEC"

Open **draft**, created 2025-10-06; nothing has touched it since the day it
was opened. One commit `4b6d8388eb`. Body describes it as a "Preview of how
we can depend on a separate SimulationSupport repo".

7 files, **+6 −278**:
`pyproject.toml` (+1−1), `support/Pipelines/Bbh/InitialData.py` (+2−2),
`support/Pipelines/EccentricityControl/EccentricityControlParams.py` (+1−5),
`support/Pipelines/EccentricityControl/InitialOrbitalParameters.py`
(**deleted**, −192), `support/Python/requirements.txt` (+2),
`tests/support/Pipelines/EccentricityControl/CMakeLists.txt` (−6),
`tests/.../Test_InitialOrbitalParameters.py` (**deleted**, −72).

Two imports replace the SpEC ones:

```python
# support/Pipelines/Bbh/InitialData.py
from SimulationSupport.EccentricityControl.InitialOrbitalParameters import (
    initial_orbital_parameters,
)
# support/Pipelines/EccentricityControl/EccentricityControlParams.py
from SimulationSupport.EccentricityControl.OmegaDotEccRemoval import (
    ComputeOmegaAndDerivsFromFile, FindTmin, performAllFits,
)
```

**Import 1 resolves. Import 2 does not exist.** SimulationSupport `main` has
no `OmegaDotEccRemoval` module — the package contains only
`EccentricityControl/{InitialOrbitalParameters,ZeroEccParamsFromPN}.py`,
`EccentricityControl/Examples/`, and `gpr/`. That port never happened.

**So PR #6890 splits cleanly in two:** its *initial-orbital-parameters* half
(this issue) is complete and correct against today's SimulationSupport; its
*eccentricity-measurement* half (#7416) is blocked on a port into
SimulationSupport that nobody has done.

Dependency declaration — a bare VCS line appended to
`support/Python/requirements.txt`:

```
# SimulationSupport: shared routines for SpEC and SpECTRE
git+https://github.com/sxs-collaboration/SimulationSupport.git
```

No PEP 508 `Name @` prefix, no tag, no commit pin — it tracks `main` HEAD,
contradicting the SimulationSupport docs' own pinned-hash workflow. No
`setup.py`, `environment.yaml`, or conda recipe touched. `pyproject.toml`
only adds `"SimulationSupport"` to isort's `known_first_party`.

**Failing checks** (run 18268519179): 19 FAILURE — `Clang-tidy (Release)`,
all 8 Linux unit-test configs, both macOS configs, all 4 `Archs`,
`Documentation`. `Commits` and `Files and formatting` passed;
`Clang-tidy (Debug)` cancelled.

The error text is **not recoverable** — job logs return HTTP 410 (expired)
and annotations carry only `"Process completed with exit code 1."` What the
surviving data does establish: every container job died in **76–105 s**, far
short of a real build; `develop` runs the same day and other fork PRs that
day succeeded, so this was not a CI outage. Only four jobs run
`pip install -r support/Python/requirements.txt` (the two macOS jobs and the
two matrix entries setting `PYTHON_VERSION`), and both macOS jobs failed —
consistent with the new `git+https://…` line breaking pip. **The ~90 s
container-job failures are unexplained by surviving data.** Re-pushing the
branch would produce fresh logs and settle it.

**Not rebased:** `mergeable=false`, `mergeable_state="dirty"` (real
conflicts). Merge base dated 2025-10-03; branch is **1 ahead, 1285 behind**
develop.

### PR #6224 — "Use PostNewtonian.jl for initial orbital parameters"

Open (not draft), created 2024-08-15, last activity 2025-05-13. **Context,
not the plan.**

Approach: replaces the SpEC kernels with two helpers driven through the
`sxs` package's Julia bridge — `omega_and_adot(r, q, chiA, chiB)` building a
`PostNewtonian.BBH` state vector, and
`num_orbits_and_time_to_merger(q, chiA, chiB, omega0)` using
`sxs.julia.PNWaveform`. **SpEC's `rPrime0` gauge parameter disappears.** The
outer Nelder-Mead control flow is unchanged. Adds `sxs >= 2024.0.3` to
requirements; Julia is pulled in transitively and downloaded on first use.
Moves the test **out** of the `if (SpEC_FOUND)` guard so it runs
unconditionally.

**It changes the numbers.** At `separation=16.0`: `Omega_0`
0.014474280975952748 → 0.014454484323416913; `adot_0`
−4.117670632867514e−05 → −4.236562633362394e−05. At `NumOrbits=20`, `D_0`
16.042 → 15.711. In-code comment claims agreement with SpEC "up to 2.5 PN
order, as tested by Mike Boyle" (moble/PostNewtonian.jl issue #41).

**Where it stalled:** changes requested 2024-11-01 with a **deployment**
blocker, not a code one — *"Currently only HPC has a sufficient version of
`sxs`. Ocean and mbot will need to be updated"* — plus three inline comments
(undocumented "magic 1s and 0s" in the BBH state vector; "Why replace?
Please add some docs"; a JEMALLOC/CLI question) and "Also needs a rebase".
Rebased and marked "ready for review" 2025-02-27, pushed again 2025-05-13;
no re-review was ever submitted and the three inline comments were never
answered. Not rebased now: `mergeable_state="dirty"`, **2 ahead, 1933
behind** develop.

**Relationship to the plan.** #6224 and the SimulationSupport plan are
*alternatives* for the same slot, and they are not compatible as written:
#6224 changes the PN numbers, SimulationSupport preserves them exactly. If
SimulationSupport is the plan, #6224 should be closed or explicitly
re-scoped to "improve the PN kernels *inside* SimulationSupport" — otherwise
the same argument gets re-litigated later against the GPR baseline, which is
defined as a residual *on top of the SpEC PN guess*.

### Dedupe

Existing trackers for this exact topic — **this issue is the canonical
one**:

| # | kind/state | relevance |
|---|---|---|
| 5933 | PR merged | "BBH ID: compute initial orbital params from PN" — created `InitialOrbitalParameters.py` and the SpEC dependency this issue removes |
| 5937 | issue OPEN | "Initial orbital parameters and eccentricity control" — the older, broader version of this issue; lists improvement options (EOB-based, varpro fits, extend PN, BNS/BHNS) |
| 5892 | issue OPEN | "Add CLI that runs SpEC ecc reduction script" — explicitly anticipates replacing the SpEC script later |
| 6224 | PR open | above |
| 6890 | PR draft | above |
| 6467 | PR merged | "Check SpEC version when importing" — added the `check_spec_import` guard both PRs delete |
| 6187 | PR merged | "Pass spec pythonpath directly to spectre" — the `SPEC_ROOT` mechanism |
| 7448 | issue OPEN | "Publish spectre Py package on pip" — packaging counterpart |
| 6366 / 3608 | PR merged | Python bindings and the C++ `BinaryTrajectories` PN class — a separate, leading-order in-tree PN implementation used only for test trajectories; not a candidate to replace `ZeroEccParamsFromPN` |

No issue or PR in the spectre repo mentions GPR, Gaussian processes, or
surrogates — that work exists only in SimulationSupport.

## Gap statement

1. SpECTRE's PN initial guesses are SpEC's, imported at runtime
   (`InitialOrbitalParameters.py:114-115`), behind a build-time
   `-D SPEC_ROOT` (`cmake/SetupSpec.cmake:4-11`).
2. Because SpEC is absent from CI, the whole path is untested there
   (`tests/support/Pipelines/EccentricityControl/CMakeLists.txt:4`).
3. The SpEC kernels themselves carry two acknowledged deficiencies that a
   port should not inherit: the T4 PN coefficients are "taken from Triton,
   should be updated" (`ZeroEccParamsFromPN.py:105`) and the integrator is
   LSODA via `odeint` rather than `solve_ivp`
   (`InitialOrbitalParameters.py:110-113`).
4. Only zero eccentricity is supported
   (`InitialOrbitalParameters.py:85-88`), although `TargetParams` already
   carries `MeanAnomalyFraction`
   (`support/Pipelines/Bbh/InitialData.py:34`) that nothing reads.
5. On the SimulationSupport side, **the function is ready and the packaging
   is not**: no PyPI release, no tags, `version = "0.1.0"` frozen, and
   `torch`+`gpytorch`+`sxs` as unconditional dependencies. PR #6890 depends
   on an unpinned `git+https://…` URL, contradicting SimulationSupport's own
   documented pinned-hash workflow (`docs/index.rst:28-31`).
6. The GPR-fitted guesses are merged in SimulationSupport but **not wired
   into `initial_orbital_parameters`**, and nothing proposes the wiring.

## What is actually blocking, in order

1. **Packaging decision** — how SpECTRE depends on SimulationSupport
   (pinned hash vs tag vs PyPI release), and whether `torch`/`gpytorch` are
   acceptable as unconditional SpECTRE dependencies or need to move to an
   extra. This is a SimulationSupport-side change and it blocks everything
   else.
2. **Rebase and re-run PR #6890's initial-orbital-parameters half** — it is
   1285 commits behind with real conflicts, and its CI failures cannot be
   diagnosed from expired logs. Splitting it so it does *not* also try to
   import the non-existent `OmegaDotEccRemoval` would make it mergeable.
3. **Decide #6224's fate** — close, or re-scope as "improve the PN kernels
   inside SimulationSupport". Leaving it open invites the same discussion
   again later.
4. GPR wiring — a separate, later step, unblocked by any of the above.

---

🤖 Drafted with [Claude Code](https://claude.com/claude-code) as the feature-parity campaign survey agent; reviewed and posted by a human.
