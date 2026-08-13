# Survey: orbit-based triggers `EveryNOrbits` / `NTimesPerOrbit`

Feature-parity survey evidence. Revisions: SpEC @ `5f8f5375ca`, SpECTRE
`develop` @ `4d43624d64` (2026-08-13). All `file:line` references are to
these revisions.

## What SpEC has

- Exactly **one** orbit-based trigger:
  `Evolution/EvolutionObservers/FractionOfOrbit.cpp:34-162` — a
  stateful, checkpointed observer that tracks the swept angle of a
  mapped reference point and fires every `2π·frac`.
- Hard cap `frac <= 0.25` (`FractionOfOrbit.cpp:44-45`): at least 4
  triggers per orbit. **"Every N orbits" is not expressible in SpEC —
  `EveryNOrbits` is a SpECTRE-only requirement.**
- Production use: wave extraction for all bound-orbit BBH runs at
  `RatchetingObservationsPerOrbit = 400`
  (`DoMultipleRuns.input:211-215`). "Ratcheting" makes the observation
  period the running minimum of past periods
  (`FractionOfOrbit.cpp:147-158`), so the cadence never coarsens as
  the orbit speeds up.
- Negative result: grep over `Evolution/`, `Utils/`, `Observers/`,
  `ComputeItems/` finds no other orbit-count machinery.

## What SpECTRE has

- **No orbit-based trigger of either kind.** The wave-extraction
  cadence in `Inspiral.yaml` is `Slabs`/`TimeCompares` only — it does
  not track the orbital frequency through the inspiral.
- The building blocks exist:
  - `SeparationLessThan`
    (`src/Evolution/Triggers/SeparationLessThan.hpp:75-127`) is the
    precedent for a trigger reading `domain::Tags::FunctionsOfTime`.
  - **The orbital phase is available in closed form**:
    `QuaternionFunctionOfTime` stores the rotation as the derivative
    of an angle `PiecewisePolynomial` and exposes `angle_func(t)` and
    derivatives (`QuaternionFunctionOfTime.hpp:68,148-160`). SpEC
    accumulates the angle only because it lacks this — a SpECTRE
    trigger can be **stateless**, which is also required:
    `Trigger::operator()` is `const` and evaluated per element.
- Caveats the design must handle:
  - `angle_func` is not on the `FunctionOfTime` base class (the base
    `func*` return the quaternion).
  - The initial angles are documented as arbitrary
    (`QuaternionFunctionOfTime.hpp:45-48`) — a phase-counting trigger
    must difference against a reference.
  - The stored angle is a 3-vector — precession needs a phase
    definition.

## Prior art — read before starting

**PR #6009 "Add Fraction of Orbit Trigger"** (open, stale since
2025-08-12) is a substantially complete attempt: trigger + test +
accumulated-angle support + BBH registration. The author's own note:

> there's something messed up in the logic of when the next check time
> should be. This causes an issue where the trigger never actually
> triggers.

So the open work is concentrated in the dense-trigger
`next_check_time` — addressed concretely in the design below. Note
`angle_func` exists on develop today; the PR's
`QuaternionFunctionOfTime` diff needs re-checking before reuse.

Related: **#5938 "Stop inspiral after fixed number of orbits"** is the
use case #6009 was written against — #5938 the use case, this issue the
mechanism, #6009 should have its home here. Pattern references:
PR #5150 (`SeparationLessThan`), #6409 (`InsideHorizon`), #2983 (dense
triggers). Nothing else covers either trigger.

**Scope note**: wiring the new trigger into `Inspiral.yaml`'s
observation events is part of the deliverable, not a follow-up — SpEC's
400/orbit *is* the production wave-extraction cadence.

## Proposed design

Two **stateless** trigger classes reading the cumulative orbital phase
`Φ(t)` from `QuaternionFunctionOfTime::angle_func`:

- `NTimesPerOrbit` (SpEC's `FractionOfOrbit`, without the 0.25 cap):
  fires when `Φ(t)` crosses the next multiple of `2π/N`;
- `EveryNOrbits` (the SpECTRE-only half): fires on multiples of `2π·N`.

Implementation shape:

- `src/Evolution/Triggers/{NTimesPerOrbit,EveryNOrbits}.{hpp,cpp}`,
  starting from PR #6009's `FractionOfOrbit` (renamed; its
  `QuaternionFunctionOfTime` diff re-checked against develop).
- Phase access: `dynamic_cast` the rotation function of time to
  `QuaternionFunctionOfTime` inside the trigger (pattern:
  `SeparationLessThan`). The trigger uses `Φ(t) − Φ(t_initial)` so the
  documented arbitrariness of the initial angles drops out.
- Dense-trigger variant for wave extraction: `next_check_time` inverts
  `Φ(t) = 2π k / N` by a scalar root-find (TOMS 748 on the angle
  `PiecewisePolynomial`) — exactly the defect that stalled #6009, now
  with a concrete fix.
- Registration in `EvolveGhBinaryBlackHole`; `Inspiral.yaml`'s
  wave-extraction/observation events switch from fixed `Slabs`
  intervals to `NTimesPerOrbit` (default: open point 5).

**Testing / acceptance**: unit tests against analytic rotation
functions of time — constant `Ω` (trigger times exactly `k·T/N`) and a
chirping `Ω` (trigger count matches the analytic phase); a
`next_check_time` test covering the #6009 failure mode; an input-file
test exercising the dense-trigger path in the BBH executable.

## Open points to settle

1. [ ] **`angle_func` access** — `dynamic_cast` in the trigger
   (recommendation: smaller first step) vs extending the
   `FunctionOfTime` interface.
2. [ ] **Phase under precession** — z-component of the angle vector,
   projection on the instantaneous orbital angular momentum, or norm.
   Recommendation: norm — reduces to the z-component for
   non-precessing systems; document the choice.
3. [ ] **Ratcheting** — reproduce SpEC's non-increasing-period
   behaviour, or rely on `Ω(t)` being monotone for quasicircular
   inspirals. Recommendation: skip initially.
4. [ ] **PR #6009** — revive with its author or supersede explicitly —
   not reimplement silently.
5. [ ] **Default cadence** in `Inspiral.yaml` — adopt SpEC's 400/orbit
   for wave extraction? Recommendation: yes.

A follow-up comment settling these points makes this issue ready for
implementation (→ Ready).

---

🤖 drafted by: [Claude Fable 5](https://claude.com/claude-code)
(and reviewed by me 🙋)
