# Survey: orbit-based triggers `EveryNOrbits` / `NTimesPerOrbit`

Feature-parity survey evidence. Revisions: SpEC @ `5f8f5375ca`, SpECTRE
`develop` @ `4d43624d64` (2026-08-13). All `file:line` references are to
these revisions.

## What SpEC has

SpEC has exactly **one** orbit-based trigger, and it is an *N-times-per-orbit*
trigger only. There is **no** `EveryNOrbits` counterpart.

### `FractionOfOrbit` — the `NTimesPerOrbit` counterpart

`Evolution/EvolutionObservers/FractionOfOrbit.cpp:34-162`,
header `Evolution/EvolutionObservers/FractionOfOrbit.hpp`.

It is *not* a `DenseTrigger` or an `EventTrigger`; it is an
`EvolutionObserver` that owns a list of sub-observers and gates them. The
mechanism (`FractionOfOrbit.cpp:73-162`):

1. On first call it picks a reference point `P` in the **grid** frame at the
   largest grid radius on the domain, along `x̂`
   (`FractionOfOrbit.cpp:82-96`).
2. Every step it maps `P` through the named `SpatialCoordMap`
   (`FractionOfOrbit.cpp:97-102`) — i.e. through the rotation/expansion maps.
3. It computes the angle between the mapped point at the current time and at
   the last trigger, via a normalised dot product
   (`FractionOfOrbit.cpp:116-139`).
4. It triggers when `angle / mDeltaAngle - 1 > -1e-12`
   (`FractionOfOrbit.cpp:147`), where
   `mDeltaAngle = 2*pi*FractionOfOrbit` (`FractionOfOrbit.cpp:46`).

Hard constraint (`FractionOfOrbit.cpp:44-45`):

```cpp
REQUIRE(frac > 0 and frac <= 0.25,
        "FractionOfOrbit must be between 0 and 0.25, not " << frac);
```

**`FractionOfOrbit <= 0.25` means SpEC can only trigger at least 4 times per
orbit. "Every N orbits" is not expressible — `EveryNOrbits` is a
SpECTRE-only requirement with no SpEC counterpart.**

It is **stateful and checkpointed**: `mOldMappedP`, `mLastTimeTriggered`,
`mDeltaT` are saved/restored (`FractionOfOrbit.cpp:186-212`).

### The "ratcheting" variant

`RatchetingObservationsPerOrbit` (`Evolution/AddWaveExtraction.hpp:32-38`)
sets `TriggerDeltaTIsNonIncreasing = yes`
(`Evolution/AddWaveExtraction.cpp:99-101,673-676`), which makes the
observation period the running **minimum** of all past periods
(`FractionOfOrbit.cpp:147-158`) — so the observation cadence never coarsens
as the orbit speeds up.

### Where it is used in production BBH

`InputFiles/Bbh/DoMultipleRuns.input:211-215`:

```perl
Readonly::Scalar(my $WaveObservationRate,
                 ($OrbitType eq "hyperbolic" ||
                  $OrbitType eq "capture") ?
                 "DeltaT = $DeltaTObserve;" :
                 "RatchetingObservationsPerOrbit = 400;");
```

i.e. **wave extraction for bound orbits is driven at 400 observations per
orbit**, exactly the use case named in this issue. It is fed to
`GrWaveExtraction.input` via `__WaveObservationRate__`
(`DoMultipleRuns.input:1129-1135`).

Everything else in SpEC's BBH input files triggers on time, steps or chunks
(`Evolution/EventTriggers/DenseTrigger*.{hpp,cpp}`).

**Negative result (searched):** grep for `NOrbits|NumOrbits|OrbitCount|
EveryNOrbit|orbits` over `Evolution/`, `Utils/`, `Observers/`,
`ComputeItems/` (`*.hpp`, `*.cpp`) returns nothing beyond `FractionOfOrbit`
and `AddWaveExtraction`.

## What SpECTRE has today

- Base class `Trigger` — `src/ParallelAlgorithms/EventsAndTriggers/Trigger.hpp`.
  Triggers are **stateless** `operator()` calls with `argument_tags` pulled
  from the DataBox.
- Time/slab triggers: `src/Time/Triggers/`. Dense triggers:
  `src/ParallelAlgorithms/EventsAndDenseTriggers/DenseTriggers/` — only
  `Times`, `Filter`, `Or`.
- The one non-time trigger, and the model to copy:
  `src/Evolution/Triggers/SeparationLessThan.hpp:75-127` — the existing
  precedent for a trigger on a derived physical quantity; its
  `argument_tags` are `tmpl::list<Tags::Time, domain::Tags::FunctionsOfTime>`
  (`:99-105`). Used in the BBH inspiral
  (`support/Pipelines/Bbh/Inspiral.yaml:479+`).
- **The orbital phase is already available in closed form.**
  `src/Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp:68` stores the
  rotation as the derivative of an angle `PiecewisePolynomial`
  (`:32-54`) and exposes `angle_func(t)` (`:148-150`),
  `angle_func_and_deriv(t)` (`:154-156`), `angle_func_and_2_derivs(t)`
  (`:158-160`).

## Gap

SpECTRE has **no orbit-based trigger of either kind**. The wave-extraction
and observation cadence in `support/Pipelines/Bbh/Inspiral.yaml` is driven
by `Slabs`/`TimeCompares` only, so it does not track the orbital frequency
as it increases through the inspiral — unlike SpEC's
`RatchetingObservationsPerOrbit = 400`.

Concrete implementation notes from the evidence:

1. **SpECTRE can be simpler than SpEC here.** SpEC accumulates the angle
   incrementally between triggers because it has no closed-form phase.
   SpECTRE's `QuaternionFunctionOfTime::angle_func(t)` gives `Φ(t)`
   directly, so both `EveryNOrbits` and `NTimesPerOrbit` can be
   **stateless**: trigger when `floor(Φ(t) / (2π/N))` increments. This
   matters because `Trigger::operator()` is `const` and evaluated per
   element — a stateful SpEC-style trigger would be wrong in SpECTRE.
2. **`angle_func` is not on the base class.** `FunctionOfTime`
   (`src/Domain/FunctionsOfTime/FunctionOfTime.hpp`) exposes only `func*`,
   which for `QuaternionFunctionOfTime` return the **quaternion**
   (`QuaternionFunctionOfTime.hpp:119-130`). A trigger reading
   `domain::Tags::FunctionsOfTime` therefore needs a `dynamic_cast` to
   `QuaternionFunctionOfTime`, or the base interface must be extended.
   Decide which.
3. **Precession.** The stored angle is a 3-vector. For non-precessing
   systems the orbital phase is the `z` component; in general it is the
   projection on the instantaneous orbital angular momentum. SpEC sidesteps
   this by using the angle between successive *mapped points*, which is
   automatically the physical swept angle. Pick a definition.
4. **Phase offset.** `QuaternionFunctionOfTime.hpp:45-48` documents "The
   initial rotation angles passed to the angle `PiecewisePolynomial` don't
   matter as we never actually use the angles themselves." A phase-counting
   trigger *does* use them, so the trigger must subtract `Φ(t_0)`, or the
   initial-angle convention must be pinned down.
5. **Dense-trigger variant.** If the trigger has to drive dense output
   (wave extraction), `DenseTrigger::next_check_time` must invert
   `Φ(t) = 2πk/N`. `Φ` is a `PiecewisePolynomial`, so this is a scalar
   root-find, not closed form.
6. **Ratcheting.** SpEC's `TriggerDeltaTIsNonIncreasing` behaviour
   (`FractionOfOrbit.cpp:147-158`) is a separate, *stateful* feature. It is
   the option actually used in production. Decide whether to reproduce it; a
   stateless equivalent is "also trigger every `min` over the analytic
   `2π/(N·Ω(t))`", which `angle_func_and_deriv` supplies since `Ω` is
   monotonically increasing for a quasicircular inspiral.

## Prior art — read this before starting

**PR #6009 "Add Fraction of Orbit Trigger" (open, stale since 2025-08-12) is
a substantially complete attempt at this issue.** It should be the starting
point, not a fresh implementation.

What it adds: `src/Evolution/Triggers/FractionOfOrbit.{hpp,cpp}` (+163/+57;
the name is taken directly from SpEC's class), a test (+86),
accumulated-angle support in `QuaternionFunctionOfTime` (+20/−5), and
registration in `EvolveGhBinaryBlackHole.hpp`.

The author's own note on the PR is the key fact:

> This isn't fully fleshed out yet… there's something messed up in the logic
> of when the next check time should be. This causes an issue where the
> trigger never actually triggers. I didn't have the time to debug it… The
> tests also need a bit of an update.

That failure mode is exactly design note 5 above — the dense-trigger
`next_check_time` inversion of `Φ(t) = 2πk/N`. So the open work is
concentrated in one known place.

Note that `QuaternionFunctionOfTime::angle_func` **exists on develop today**
(`QuaternionFunctionOfTime.hpp:148-150`), so part of what #6009 proposed may
since have landed by another route; the PR's `QuaternionFunctionOfTime` diff
needs re-checking against current develop before it is reused.

Related issue: **#5938 "Stop inspiral after fixed number of orbits"** (open)
— the motivating issue #6009 was written against. It is an `EveryNOrbits`
use case, so this issue and #5938 overlap: #5938 is the use case, this issue
is the mechanism, and #6009 should have its home here.

Other dedupe results:

| # | kind/state | relevance |
|---|---|---|
| 5150 | PR merged | "Add standard trigger for separation between objects" — created `SeparationLessThan`, the nearest merged analogue and the pattern to follow |
| 6409 | PR merged | "Add `InsideHorizon` trigger for worldtube excision" — another physical-quantity trigger for reference |
| 2983 | PR merged | "Add dense triggers" — the dense-trigger framework |

No other open issue or PR covers `EveryNOrbits` or `NTimesPerOrbit`.

## Scope note

The issue text says "We also need this for wave extraction, etc." — SpEC's
usage confirms that: `RatchetingObservationsPerOrbit = 400` is *the*
wave-extraction cadence for all bound-orbit BBH runs
(`DoMultipleRuns.input:211-215`). Wiring the new trigger into
`Inspiral.yaml`'s observation events is part of the deliverable, not a
follow-up.

## Proposed design

Two **stateless** trigger classes reading the cumulative orbital phase
`Φ(t)` from `QuaternionFunctionOfTime::angle_func`:

- `NTimesPerOrbit` (SpEC's `FractionOfOrbit`, without the 0.25 cap):
  fires when `Φ(t)` crosses the next multiple of `2π/N`;
- `EveryNOrbits` (the SpECTRE-only half): fires on multiples of `2π·N`.

Implementation shape:

- `src/Evolution/Triggers/{NTimesPerOrbit,EveryNOrbits}.{hpp,cpp}`,
  starting from PR #6009's `FractionOfOrbit` (renamed; its
  `QuaternionFunctionOfTime` diff re-checked against develop, where
  `angle_func` already exists).
- Phase access: `dynamic_cast` the rotation function of time to
  `QuaternionFunctionOfTime` inside the trigger (pattern:
  `SeparationLessThan`, which also reads
  `domain::Tags::FunctionsOfTime`). The trigger uses
  `Φ(t) − Φ(t_initial)` so the documented arbitrariness of the initial
  angles drops out.
- Dense-trigger variant for wave extraction: `next_check_time` inverts
  `Φ(t) = 2π k / N` by a scalar root-find (TOMS 748 on the angle
  `PiecewisePolynomial`) — this is exactly the defect that stalled
  #6009, now with a concrete fix.
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
