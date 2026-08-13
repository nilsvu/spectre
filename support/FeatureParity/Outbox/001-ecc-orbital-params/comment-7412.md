# Survey: initial orbital parameters — SpEC's `ZeroEccParamsFromPN` vs SpECTRE vs the SimulationSupport plan

Feature-parity survey evidence. Revisions: SpEC @ `5f8f5375ca`, SpECTRE
`develop` @ `4d43624d64`, `sxs-collaboration/SimulationSupport` `main`
(2026-08-13). All `file:line` references are to these revisions.

**Framing.** The plan of record is to wire
`sxs-collaboration/SimulationSupport` into **both** SpEC and SpECTRE as
the provider of initial orbital parameters, with GPR-fitted initial
guesses being added there as an alternative to the PN guesses. PR #6224
(PostNewtonian.jl) is recorded below as context, not as the plan.

## What SpEC has

- `Support/Python/ZeroEccParamsFromPN.py`: closed-form 3.5PN `Ω(r)` and
  `ṙ/r` (`:27-63`) plus a 3.5PN T4 orbit integrator (`:66-158`).
  Known deficiencies annotated in-source: T4 coefficients *"Taken from
  Triton, should be updated"* (`:105`); LSODA via `odeint`.
- It is a standalone helper, not a pipeline stage: no BBH driver calls
  it; the user pastes `Omega0`/`adot0`/`D0` into `Params.input` by
  hand.

## What SpECTRE has

- The wrapper the plan needs **already exists**:
  `initial_orbital_parameters(target_params, ...)` in
  `support/Pipelines/EccentricityControl/InitialOrbitalParameters.py:15-198`,
  with its own root-finds and `rPrime0=1.0` pinned "as in SpEC".
- Only the two physics kernels come from SpEC (`:110-115`):

  ```python
  check_spec_import()
  from ZeroEccParamsFromPN import nOrbitsAndTotalTime, omegaAndAdot
  ```

  behind the build-time `-D SPEC_ROOT` mechanism
  (`cmake/SetupSpec.cmake:4-11`).
- **Consequence: none of this runs in CI.** Every ecc-control test is
  wrapped in `if (SpEC_FOUND)`
  (`tests/support/Pipelines/EccentricityControl/CMakeLists.txt:4-17`)
  and CI has no SpEC checkout. This is the strongest argument for the
  SimulationSupport plan, independent of any physics improvement.
- Only zero eccentricity is supported (`:85-88`); `TargetParams`
  carries a `MeanAnomalyFraction` that nothing reads.

## What SimulationSupport provides today

- Its `initial_orbital_parameters` is a **near-verbatim copy of
  SpECTRE's**, carrying SpEC's `ZeroEccParamsFromPN.py` inside the
  package; its test asserts the SpEC numbers byte-identically
  (`tests/EccentricityControl/Test_InitialOrbitalParameters.py:20`).
- **Adoption is therefore a pure refactor with no numerical change**:
  delete SpECTRE's copy and import.
- **The GPR work** (Vittoria Tommasini) is real and mostly merged
  (`src/SimulationSupport/gpr/`, gpytorch/torch; trained on 89 SXS BBH
  simulations, 4 features). Two facts matter for the plan:
  - It is an **additive correction on top of the PN guess** — GPs
    trained on residuals relative to `ZeroEccParamsFromPN.omegaAndAdot`
    at `rPrime0 = 1`. Changing the PN baseline invalidates the fits.
  - It is **not wired into `initial_orbital_parameters`** — no branch
    or PR proposes the wiring; trained checkpoints are not committed.
    The refactor above will not need redoing when it lands.
- **Packaging** is not an obstacle (SpECTRE's build configs pull Python
  dependencies automatically; pip installs from a git URL). Two
  decisions remain, folded into the open points:
  - **Nothing to pin to yet**: no tags or releases; the
    SimulationSupport docs themselves prescribe a pinned commit hash
    (`docs/index.rst:28-31`).
  - `torch`/`gpytorch` are installed unconditionally although only
    `gpr/` needs them — large, platform-specific wheels, felt in CI
    images and cluster environments; `sxs` is likewise unused on
    `main`. An optional extra would keep the base install light.

## Prior art

**PR #6890 "Ecc control: depend on SimulationSupport instead of SpEC"**
(draft, untouched since 2025-10-06; 1 ahead / 1285 behind, real
conflicts):

- **Splits cleanly in two.** Its initial-orbital-parameters half is
  complete and correct against today's SimulationSupport. Its
  measurement half imports
  `SimulationSupport.EccentricityControl.OmegaDotEccRemoval` — **a
  module that does not exist**; that port never happened (tracked in
  #7416).
- Its requirements line is unpinned (tracks `main` HEAD).
- Its 19 failing checks are undiagnosable: logs expired (HTTP 410);
  container jobs died in 76–105 s while same-day `develop` runs
  passed. Re-pushing produces fresh logs and settles it.

**PR #6224 "Use PostNewtonian.jl for initial orbital parameters"**
(open since 2024-08, last push 2025-05, 1933 behind):

- **It changes the PN numbers** (at `separation=16`: `Omega_0`
  0.0144742810 → 0.0144544843; at `NumOrbits=20`: `D_0` 16.042 →
  15.711).
- Stalled on a deployment review blocker (sxs version on clusters);
  three inline review comments never answered.
- Incompatible with the plan as written — the GPR residuals are
  defined against the SpEC PN baseline. Proposal: re-scope to "improve
  the PN kernels *inside* SimulationSupport" (comment proposed on the
  PR).

**Dedupe**: #5933 (merged) created the current SpEC dependency; #5937
is the older, broader tracker (proposed narrowed to nonzero-target ecc
+ BNS/BHNS); #5892 (CLI, shipped) proposed closed. No spectre issue or
PR mentions GPR — that work exists only in SimulationSupport.

## Proposed design

**SimulationSupport side** (one small PR there — we can prepare it; a
maintainer merges):

- Move `torch`, `gpytorch` (needed only by `gpr/`) and `sxs` (unused on
  `main`) to an optional extra —
  `[project.optional-dependencies] gpr = [...]` — and guard the `gpr/`
  imports with a helpful error message.
- Create the first tag (e.g. `v0.1.0`), or skip tagging and pin by
  commit hash (open point 1).

**SpECTRE side** (one PR — the revived initial-orbital-parameters half
of PR #6890):

- Delete `support/Pipelines/EccentricityControl/InitialOrbitalParameters.py`
  (−192 lines); in `support/Pipelines/Bbh/InitialData.py` import
  `initial_orbital_parameters` from
  `SimulationSupport.EccentricityControl.InitialOrbitalParameters`.
  Byte-compatible — no numerical change, no call-site changes.
- `support/Python/requirements.txt`: add the pinned PEP 508 line
  `SimulationSupport @ git+https://github.com/sxs-collaboration/SimulationSupport.git@<pin>`.
- Replace `tests/.../Test_InitialOrbitalParameters.py` with a thin
  integration test that calls the imported function once and asserts
  today's numbers, moved **out** of the `if (SpEC_FOUND)` block — this
  path then runs in CI for the first time.
- `EccentricityControlParams.py` is **not** touched: the measurement
  side keeps its SpEC import until `OmegaDotEccRemoval` is ported to
  SimulationSupport (tracked in #7416).
- Rebase on develop; fresh CI also settles the old unexplained
  container-job failures.

**Related actions**: PR #6224 re-scope as above. GPR wiring into
`initial_orbital_parameters` is a separate, later SimulationSupport-side
step; nothing above needs redoing when it lands.

**Testing / acceptance**: the integration test runs unconditionally in
CI and reproduces today's numbers
(`[16.0, 0.014474280975952748, -4.117670632867514e-05]` at
`separation=16`); `spectre bbh generate-id` without explicit orbital
parameters works in a clean environment built from `requirements.txt`.

## Open points to settle

1. [ ] **Pin mechanism** — commit hash in `requirements.txt` (per the
   SimulationSupport docs) vs first tag. Recommendation: hash now,
   switch to tags once SimulationSupport starts releasing.
2. [ ] **Extras split** — move `torch`/`gpytorch`/`sxs` to a
   `SimulationSupport[gpr]` extra? Recommendation: yes — small upstream
   change, keeps SpECTRE environments light.
3. [ ] **Upstream driver** — who merges the SimulationSupport-side
   changes (we can prepare the PRs).
4. [ ] **SpECTRE-side test** — drop `Test_InitialOrbitalParameters`
   entirely (assertions live upstream) vs keep the thin integration
   test. Recommendation: keep it, so SpECTRE CI exercises the import.
5. [ ] **PR #6224** — confirm the re-scope, or close.

A follow-up comment settling these points makes this issue ready for
implementation (→ Ready).

---

🤖 drafted by: [Claude Fable 5](https://claude.com/claude-code)
(and reviewed by me 🙋)
