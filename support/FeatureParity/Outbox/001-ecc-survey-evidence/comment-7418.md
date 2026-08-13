# Survey: compare gauge params etc. with SpEC — enumeration and pointers

Feature-parity survey evidence. Revisions: SpEC @ `5f8f5375ca`, SpECTRE
`develop` @ `4d43624d64` (2026-08-13). All `file:line` references are to
these revisions.

Scope as surveyed: **enumerate the parameter sets that must be compared and
give both sides' locations.** A full numeric comparison is deliberately out
of scope here; where the two codes already differ visibly and the
difference is documented in the source, that is recorded.

**Shape of this issue (proposal reflected in the retitle):** the evidence
shows two different jobs under one title — (a) *document the intentional
differences*: small, bounded, mostly already done in code comments, with
exactly one undocumented difference found (gauge roll-on, Set 1); and
(b) *establish that structurally different schemes are equivalent in
accuracy* (time stepping, filtering, AMR): a validation-run programme,
split out as its own issue.

## Where the parameters live

| | SpEC | SpECTRE |
|---|---|---|
| Driver that computes values | `InputFiles/Bbh/DoMultipleRuns.input` (1162 lines) | `support/Pipelines/Bbh/Inspiral.py` |
| Template consuming them | `InputFiles/Bbh/*.input` (30 files) | `support/Pipelines/Bbh/Inspiral.yaml` (722 lines) |
| Values from initial data | `ID/EvID/ID_Params.perl`, read at `DoMultipleRuns.input:154-161` | ID input file metadata, read at `Inspiral.py:112-303`; SpEC-ID path at `:306-324,327-414` |

## Set 1 — Gauge source function (damped harmonic)

**SpEC:** `InputFiles/Bbh/GaugeItems.input` (80 lines), parameters from
`DoMultipleRuns.input:1088-1095`; implementation
`EvolutionSystems/GeneralizedHarmonic/DampedHarmonicGaugeItems.cpp`.
Key values: `SecondaryWeightRmax = 100` (`GaugeItems.input:13`),
roll-off/roll-on timescales `$wRolloff = $wRollon = 50.0`
(`DoMultipleRuns.input:624-631`), initial gauge from ID
(`GaugeItems.input:8-9,27-29`). The secondary weight is a radial Gaussian
(`DampedHarmonicGaugeItems.cpp:436-461`);
`W = log²(√detg / N) · (1 − R) · Z` (`:327-371`).

**SpECTRE:** `src/Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/
DampedHarmonic.hpp:209-237`, roll-on variant
`DhGaugeParameters.hpp:31-90`. Values at
`support/Pipelines/Bbh/Inspiral.yaml:256-262`:

```yaml
      DampedHarmonic:
        SpatialDecayWidth: 17.0152695482514 # From SpEC run: 100.0/sqrt(34.54)
        Amplitudes: [1.0, 0.0, 1.0]         # From SpEC run: damped harmonic
        Exponents: [2, 2, 2]                # From SpEC run
```

**Already reconciled:** the width conversion is documented in the template
comment (`Inspiral.yaml:249-255`), citing SpEC's
`DampedHarmonicGaugeItems.cpp` line 463 — `100.0/sqrt(34.54) = 17.0153`.

**The one undocumented difference (the concrete finding of this issue):**
SpECTRE uses `DhGaugeParameters<false>` — it starts **directly in damped
harmonic**. SpEC rolls the initial-data gauge off and the damped-harmonic
coefficients on over 50 M (`GaugeItems.input:20-25`,
`DoMultipleRuns.input:621-632`), and
`$TimeAtWhichInitGaugeIsRolledOff = StartTime + 3*50`
(`DoMultipleRuns.input:639`) gates when shell radii may change. SpECTRE has
the roll-on class (`DhGaugeParameters.hpp:31`) and does not use it in the
BBH pipeline. **This has closed prior art that must be read before
re-opening the question** (see dedupe below) — the answer may be that the
roll-on was deliberately dropped, in which case the job is only to document
it.

## Set 2 — Constraint damping (γ0, γ1, γ2)

**SpEC:** `InputFiles/Bbh/ConstraintDamping.input`, amplitudes and widths
from `DoMultipleRuns.input:187-203`: `Amp{A,B} = 4/m`, `Width{A,B} = 7·m`,
`AmpOrigin = 0.075/M`, `WidthOrigin = 2.5·d`, `Asymptotic = 0.001/M`; all
widths scaled by `S = InvExpansionFactor` (`ConstraintDamping.input:8-12`).
γ1: `0.999*(W-1)` with `W` a radial `GeneralizedGaussian(Width = 10*$d)`
(`ConstraintDamping.input:55-66`).

**SpECTRE:** `src/PointwiseFunctions/ConstraintDamping/` —
`TimeDependentTripleGaussian.hpp:56-118`, `GaussianPlusConstant.hpp:41-63`.
Values at `Inspiral.yaml:263-285`, computed in `Inspiral.py:90-109`.

**Form is identical — verified algebraically, not assumed:**
`TimeDependentTripleGaussian.cpp:109-130` computes
`amplitude · exp(-r²·(inverse_width·expansion_factor)²)`, i.e.
`exp(-r²/(width/a)²)`, matching SpEC's `exp(-dist²/(Width·S)²)` with
`S = 1/a`. γ1's `GaussianPlusConstant` with `Constant: -0.999, Amplitude:
0.999, Width: 10·separation` (`Inspiral.yaml:279-284`, `Inspiral.py:108`)
is algebraically SpEC's `0.999*(W-1)`.

**Known differences, already documented in the SpECTRE source:**

| Parameter | SpEC | SpECTRE | SpECTRE's stated reason |
|---|---|---|---|
| `Gamma0Constant` (asymptotic) | `0.001/M` (`DoMultipleRuns.input:199`) | `0.01/M` (`Inspiral.py:99`) | *"we found 0.01 produces smaller constraints violations in the envelope/outer shell region"* — `Inspiral.py:97-98` |
| `Gamma0OriginAmplitude` | `0.075/M` (`DoMultipleRuns.input:196`) | `0.75/M` (`Inspiral.py:106`) | *"we found that 0.75 produces a smaller burst of constraints from junk radiation"* — `Inspiral.py:104-105` |

Matching: `4/m` amplitudes, `7·m` widths, `2.5·d` origin width, `10·d` γ1
width.

**Not in SpECTRE:** SpEC's `$MoreConstraintDampingInTheOuterSubdomains`
flag (`DoMultipleRuns.input:79-84,201-203`), which multiplies the
asymptotic value by 10 for runs starting closer than 20 M separation.

## Set 3 — Control system

**SpEC:** `DoMultipleRuns.input:271-336` (values),
`InputFiles/Bbh/GrStateChangers.input` (structure). Key values:
`$TaverageFac = 0.25`, `$IncreaseFactor = 1.01`, `$DecreaseFactor = 0.98`,
`$coef = 0.2` (0.1 if either spin > 0.9), `$Tdamping = coef·M`,
`$TdampingShapeA/B = 5·$Tdamping`, `$ThresholdBase = 2e-3` (2e-4 high
spin), `$MaxDampTime = 20` (10 high spin).

**SpECTRE:** `Inspiral.py:36-87`, consumed at `Inspiral.yaml:611-718`.
`AverageTimescaleFraction: 0.25`, `IncreaseFactor: 1.01`,
`DecreaseFactor: 0.98` match; the high-spin branch (`Inspiral.py:45-52`)
matches SpEC's `:265-269`.

**Two items are *not* like-for-like and need derivation, not lookup:**
- `Controller UpdateFraction: 0.3` (comment *"Changed UpdateFraction from
  0.03 to 0.3 to increase run speed"*, `Inspiral.yaml:621-622`; size
  controller `0.2` at `:674-675`) vs SpEC's
  `$TstateOverTdamp`/`$MeasureFractionOfChunk` machinery.
- SpEC's char-speed (`AhSpeed`) control systems
  (`GrStateChangers.input:340-360`) vs SpECTRE's `SizeA`/`SizeB` control
  errors (`Inspiral.yaml:672-714`) — structurally different; needs a
  mapping table before any numeric comparison.

Shape-map initial values also differ (analytic Kerr vs measured ID
horizon) — evidence in the #7417 survey comment, split out as its own
issue.

## Set 4 — Time stepping and error control

| | SpEC | SpECTRE |
|---|---|---|
| Integrator | `DormandPrince5` (single-step RK) — `Evolution.input:220` | `AdamsMoultonPcMonotonic` order 4 (multistep PC) — `Inspiral.yaml:242-244` |
| Stepping mode | global adaptive, `AdaptiveDense` — `Evolution.input:213` | local time stepping, `LocalTimeStepping: Conservative` — `Inspiral.yaml:226` |
| Step controller | `ProportionalIntegral` — `Evolution.input:214-221` | `LtsStepChoosers` (`LimitIncrease Factor 2`, `PreventRapidIncrease`, `ErrorControl`) — `Inspiral.yaml:229-240` |
| Tolerance | `ODETolerance = 1e-8` — `DoMultipleRuns.input:830` | `AbsoluteTolerance 1e-10, RelativeTolerance 1e-8` — `Inspiral.yaml:233-239`, comment *"100x smaller timestep tolerances reduced the noise in the constraints significantly"* |
| Initial step | `min(1e-3, 0.1·min(TdampingA,TdampingB))` — `DoMultipleRuns.input:300` | `0.0002` hard-coded — `Inspiral.yaml:220` |
| Minimum step | `1e-5` — `Evolution.input:217` | `1e-7` — `Inspiral.yaml:221` |
| Order control | n/a (fixed RK5) | `VariableOrderAlgorithm GoalOrder: 4` — `Inspiral.yaml:227-228` |

**Structurally different schemes** → validation programme (split issue),
not parameter matching. The hard-coded initial step is separately a defect
of the PBJ branch (#7413).

## Set 5 — Filtering / spatial discretization

- **SpEC:** exponential filters configured inside the Fosh mover; BBH-level
  knobs visible in `Evolution.input:258-260` (`InternalBcFilter = true`)
  and the filtered-copy step `Evolution.input:16-19`. The filter
  definitions themselves live in domain/subdomain input **not reached by
  this survey** (searched `InputFiles/Bbh/*.input` for `Filter`).
- **SpECTRE:** `Inspiral.yaml:287-303` — `AveragedUpwindPenalty` boundary
  correction, `StrongInertial` formulation, `GaussLobatto` quadrature,
  `Hypercube` filter with `HalfPower: 420` (comment `:293-294`: chosen "to
  filter only the last term up to N = 19"), `VolumeFilterOnSubstep: true`.

Different discretizations (multi-domain spectral vs DG). Comparison is
about *effective dissipation* → validation programme.

## Set 6 — Resolution, AMR, and tolerances

- **SpEC:** `AmrTolerances.input` from `DoMultipleRuns.input:823-843`:
  `TruncationErrorMax = 0.000216536 · 4^(-k)` and derived values. Real AMR:
  `AmrDriver.input`, cadence `TriggerEveryNChunks`
  (`DoMultipleRuns.input:654,699-719`).
- **SpECTRE:** Lev is pure p-refinement, `polynomial_order = 7 + lev`,
  `refinement_level = 1` (`Inspiral.py:26-33`, comment *"To be replaced
  once AMR is used"*). The `Amr:` block (`Inspiral.yaml:313-324`) has an
  **empty `Criteria:`** — AMR is configured but inert.

**Not comparable today.** Structural gap; belongs to the AMR work, not to
this issue.

## Set 7 — Domain, outer boundary, wave extraction

- Outer radius: SpEC `SpEC::AutoRmax` (`DoMultipleRuns.input:576-583`);
  SpECTRE has a TODO citing it (`Inspiral.py:227-231,378-382`).
- Outer-boundary drift speed: SpEC `SpEC::SetOuterBdrySpeed`
  (`DoMultipleRuns.input:589-590`); SpECTRE
  `AsymptoticVelocityOuterBoundary: -1.0e-6`,
  `DecayTimescaleOuterBoundary: 50.0` (`Inspiral.yaml:174-176`) — the
  `-1e-6` matches SpEC's hyperbolic/capture default
  (`DoMultipleRuns.input:500,534`) but SpEC computes it for bound orbits.
- Excision radius: SpEC `rExc = sqrt(ExtrFrac)·rInitAh`
  (`DoMultipleRuns.input:381-387`); SpECTRE `excision_radius_factor_a` 1.0
  or 1.0385 (`Inspiral.py:174-176`), comment about SpEC's 0.97 factor at
  `:361-363`.
- Wave-extraction cadence: SpEC `RatchetingObservationsPerOrbit = 400` vs
  SpECTRE fixed-interval triggers — this is **#7415**, not this issue.

## Set 8 — Boundary conditions and observation

- SpEC `InputFiles/Bbh/GrBoundaryConditions.input`, CoM options at
  `DoMultipleRuns.input:1029-1035`.
- SpECTRE: boundary conditions in the `Inspiral.yaml` domain block; CCE via
  `BondiSachsInterpolation` on a dense trigger with `Interval: 0.1`
  (`Inspiral.yaml:506-515`, comment *"An interval of 0.1 was found to work
  well in SpEC"*).
- SpEC observation cadence `$DeltaTObserve = 0.5`,
  `$DeltaTObserveVolumeDump = 10·$DeltaTObserve`
  (`DoMultipleRuns.input:209-210`).

## Conclusion

- Gauge and constraint damping are **already deliberately reconciled**,
  with the deviations and their reasons written into the SpECTRE source
  (`Inspiral.yaml:246-255`, `Inspiral.py:97-98`, `:104-105`). Those three
  comments are the entirety of the existing documentation — collecting them
  plus the roll-on difference into one document is the bounded
  documentation task this issue keeps.
- Control system values largely match; the two that do not
  (`UpdateFraction`, char-speed↔size mapping) need a derivation before any
  number can be compared.
- Time stepping, filtering and AMR are structurally different schemes;
  "consistency with SpEC" is not the right frame for them — accuracy at
  fixed cost is, which needs runs (the split validation issue; #5133 is the
  setup issue any such comparison depends on).

## Prior-art dedupe

**The gauge roll-on finding has direct, closed prior art.** PR **#1627
"Improve damped harmonic gauge, add ability to reproduce SpEC"** was closed
in 2022, together with issues **#1516** and **#1515** (both closed
2022-12-04). That cluster is exactly the roll-on difference identified
above. **Read why it was closed before re-opening the question.**

| # | kind/state | relevance |
|---|---|---|
| **1627** | **PR closed** | **"Improve damped harmonic gauge, add ability to reproduce SpEC"** (2022) |
| **1516** | **issue CLOSED** | **"Change damped harmonic gauge rolloffs to be able to reproduce SpEC"** |
| **1515** | **issue CLOSED** | **"Reduce initial-gauge dependency of damped harmonic gauge"** |
| **5133** | **issue OPEN** | **"Choose BBH configuration for comparison with SpEC"** — the setup issue for any numeric comparison; the validation half depends on it |
| 6798 | PR merged | "Change ringdown gauge parameters to match inspiral" (2025-08) — direct precedent for the "consistent with inspiral/merger" half of this issue's text |
| 2508 | PR merged | "Add options for damped harmonic gauge amplitudes, exponents" |
| 5504 | PR merged | "Fix bug in damped harmonic gauge" |
| 2494 | PR merged | "Add GH constraint damping functions with options" |
| 6390 | PR merged | "Constraint damping gaussian's amplitudes are exchanged" — a bug fix worth knowing when comparing amplitudes |
| 6635 / 6637 | PR merged | Relocation of constraint damping to `PointwiseFunctions` |
| 2116 | issue OPEN | "Add input file control for constraint damping parameters" (2020) |
| 1811 | issue OPEN | "Add extra constraint damping parameter to generalized harmonic system" (2019) |
