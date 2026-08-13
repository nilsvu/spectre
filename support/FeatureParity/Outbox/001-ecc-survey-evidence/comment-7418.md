# Survey: compare gauge params etc. with SpEC — enumeration and pointers

Feature-parity survey evidence. Revisions: SpEC @ `5f8f5375ca`, SpECTRE
`develop` @ `4d43624d64` (2026-08-13). All `file:line` references are to
these revisions.

Scope: **enumerate the parameter sets to compare and give both sides'
locations**; a full numeric comparison is deliberately out of scope.
The evidence shows two jobs under one title (reflected in the retitle):

- (a) **document the intentional differences** — small, bounded, mostly
  already in code comments, with exactly one undocumented difference
  found (gauge roll-on, Set 1);
- (b) **establish that structurally different schemes are equivalent in
  accuracy** (time stepping, filtering, AMR) — a validation-run
  programme, split out as its own issue.

Where the parameters live: SpEC — `InputFiles/Bbh/DoMultipleRuns.input`
computes values consumed by the `InputFiles/Bbh/*.input` templates;
SpECTRE — `support/Pipelines/Bbh/Inspiral.py` fills `Inspiral.yaml`.

## Set 1 — Gauge source function (damped harmonic)

- **Already reconciled and documented**: SpECTRE's
  `SpatialDecayWidth: 17.0152695482514` carries the conversion
  `100.0/sqrt(34.54)` from SpEC's `SecondaryWeightRmax = 100`, with the
  SpEC source line cited in the template comment
  (`Inspiral.yaml:249-262` vs `GaugeItems.input:13`).
- **The one undocumented difference (the concrete finding of this
  issue):** SpECTRE starts **directly in damped harmonic**
  (`DhGaugeParameters<false>`). SpEC rolls the ID gauge off and the
  damped-harmonic coefficients on over 50 M (`GaugeItems.input:20-25`,
  `DoMultipleRuns.input:621-632`), and
  `$TimeAtWhichInitGaugeIsRolledOff` gates when shell radii may change
  (`:639`). SpECTRE has the roll-on class (`DhGaugeParameters.hpp:31`)
  unused in the BBH pipeline.
- **Closed prior art must be read before re-opening this** (see
  below) — the roll-on may have been deliberately dropped, in which
  case the job is only to document it.

## Set 2 — Constraint damping (γ0, γ1, γ2)

- **Form is identical — verified algebraically, not assumed**:
  `TimeDependentTripleGaussian.cpp:109-130` reproduces SpEC's
  `Amp·exp(-dist²/(Width·S)²)` including the inverse-expansion scaling;
  γ1's `GaussianPlusConstant` (`Constant: -0.999, Amplitude: 0.999`)
  is algebraically SpEC's `0.999·(W−1)`
  (`ConstraintDamping.input:55-66` vs `Inspiral.yaml:279-284`).
- **Matching values**: `4/m` amplitudes, `7·m` widths, `2.5·d` origin
  width, `10·d` γ1 width.
- **Documented intentional differences** (reasons in `Inspiral.py`
  comments):

  | Parameter | SpEC | SpECTRE | stated reason |
  |---|---|---|---|
  | `Gamma0Constant` | `0.001/M` | `0.01/M` | smaller constraint violations in envelope/outer shell (`Inspiral.py:97-99`) |
  | `Gamma0OriginAmplitude` | `0.075/M` | `0.75/M` | smaller junk-radiation constraint burst (`Inspiral.py:104-106`) |

- **Not in SpECTRE**: `$MoreConstraintDampingInTheOuterSubdomains`
  (`DoMultipleRuns.input:79-84,201-203`) — ×10 asymptotic value for
  initial separations < 20 M.

## Set 3 — Control system

- **Values largely match** (`Inspiral.py:36-87` vs
  `DoMultipleRuns.input:271-336`): averaging fraction 0.25,
  increase/decrease factors 1.01/0.98, and the high-spin branch
  (damping base 0.1/0.2, thresholds 2e-4/2e-3, max timescale 10/20).
- **Two items are not like-for-like and need derivation, not lookup**:
  - `Controller UpdateFraction: 0.3` (template comment: raised from
    0.03 for speed) vs SpEC's
    `$TstateOverTdamp`/`$MeasureFractionOfChunk` machinery;
  - SpEC's char-speed (`AhSpeed`) control systems
    (`GrStateChangers.input:340-360`) vs SpECTRE's `SizeA/B` control
    errors (`Inspiral.yaml:672-714`).
- Shape-map initial values differ too — tracked in the #7417 survey
  comment and its split issue.

## Set 4 — Time stepping and error control

| | SpEC | SpECTRE |
|---|---|---|
| Integrator | `DormandPrince5` (single-step RK) — `Evolution.input:220` | `AdamsMoultonPcMonotonic` order 4 (multistep PC) — `Inspiral.yaml:242-244` |
| Stepping | global adaptive `AdaptiveDense` | local time stepping, `Conservative` |
| Tolerance | `ODETolerance = 1e-8` | `1e-10`/`1e-8` abs/rel; comment: 100× smaller "reduced the noise in the constraints significantly" |
| Initial step | `min(1e-3, 0.1·min(Tdamping))` — `DoMultipleRuns.input:300` | `0.0002` hard-coded — `Inspiral.yaml:220` (a #7413 defect) |
| Minimum step | `1e-5` | `1e-7` |

**Structurally different** → validation programme (split issue), not
parameter matching.

## Set 5 — Filtering / spatial discretization

- SpEC: exponential filters in the Fosh mover; the definitions live in
  domain/subdomain input **not reached by this survey** (searched
  `InputFiles/Bbh/*.input`; only `InternalBcFilter` knobs found,
  `Evolution.input:258-260`).
- SpECTRE: DG with `Hypercube` filter `HalfPower: 420`
  (`Inspiral.yaml:287-303`).
- Comparable quantity is effective dissipation → validation programme.

## Set 6 — Resolution / AMR

- SpEC: real AMR, `TruncationErrorMax = 0.000216536·4^(-k)`
  (`DoMultipleRuns.input:823-843`).
- SpECTRE: Levs are pure p-refinement (`Inspiral.py:26-33`, *"To be
  replaced once AMR is used"*); the `Amr:` block has an **empty
  `Criteria:`** (`Inspiral.yaml:313-324`) — configured but inert.
- Not comparable today; belongs to the AMR work, not this issue.

## Sets 7–8 — Domain, boundaries, observation (pointers for the document)

- Outer radius: SpEC computes it (`SpEC::AutoRmax`); SpECTRE has a TODO
  citing it (`Inspiral.py:227-231,378-382`).
- Outer-boundary drift speed: SpECTRE's `-1e-6` matches SpEC's
  hyperbolic/capture default, but SpEC *computes* it for bound orbits
  (`Inspiral.yaml:174-176` vs `DoMultipleRuns.input:589-590`).
- Excision radius factors: `Inspiral.py:174-176,361-363` (comment on
  SpEC's 0.97 factor).
- CoM boundary-condition options: `DoMultipleRuns.input:1029-1035`.
- CCE interpolation interval 0.1 — "found to work well in SpEC"
  (`Inspiral.yaml:506-515`).
- Wave-extraction cadence is #7415, not this issue.

## Prior art

**The gauge roll-on has direct, closed prior art**: PR **#1627**
"Improve damped harmonic gauge, add ability to reproduce SpEC" (closed
2022) with issues **#1516**/**#1515** (closed 2022-12-04) — exactly the
roll-on difference above. Read why they were closed first.

| # | state | relevance |
|---|---|---|
| **5133** | issue OPEN | "Choose BBH configuration for comparison with SpEC" — the setup issue the validation half depends on |
| 6798 | PR merged | "Change ringdown gauge parameters to match inspiral" — precedent for the inspiral/merger-consistency half |
| 2508 / 5504 / 2494 / 6390 / 6635 / 6637 | PR merged | gauge and constraint-damping options history (incl. #6390, a swapped-amplitudes bug fix worth knowing when comparing) |
| 2116 / 1811 | issue OPEN | old constraint-damping option requests |

## Proposed design

One reference page, e.g. `docs/DevGuide/BbhSpecDifferences.md` (linked
from the BBH pipeline documentation), with one table per parameter set:
parameter | SpEC value + `file:line` | SpECTRE value + `file:line` |
intentional? | reason/reference. Seeded from this survey:

- the three existing source comments (`Inspiral.yaml:249-255`,
  `Inspiral.py:97-98`, `:104-105`);
- the gauge roll-on (classification per open point 2, after reading the
  #1627/#1516/#1515 closure reasons);
- `$MoreConstraintDampingInTheOuterSubdomains` (absent in SpECTRE);
- the `tmin` 400/500 drift in the ecc-reduction run length;
- the fit-option drift (recorded and settled in the #7416 survey
  thread; the page cites the outcome).

Structural differences (time stepping, filtering, AMR) get one row each
pointing to the split validation issue, not an argument in the
document. Source comments stay where they are; the page cites them.

**Acceptance**: every difference found by this survey appears in the
table classified as "intentional (reason)" or "open (issue link)"; no
undocumented differences remain.

## Open points to settle

1. [ ] **Location** — `docs/DevGuide/` page (recommendation) vs source
   comments only.
2. [ ] **Gauge roll-on** — document as intentionally dropped (if the
   #1627 closure supports that reading) or reopen as a to-match item.
3. [ ] **`tmin` 400 (SpEC) vs 500 (SpECTRE)** in the ecc-reduction run
   length — adjudicate and record.

A follow-up comment settling these points makes this issue ready for
implementation (→ Ready).

---

🤖 drafted by: [Claude Fable 5](https://claude.com/claude-code)
