# Survey: control system initialization at run start

Feature-parity survey evidence. Revisions: SpEC @ `5f8f5375ca`, SpECTRE
`develop` @ `4d43624d64` (2026-08-13). All `file:line` references are to
these revisions. Companion: #7413 (the same state problem at a PBJ branch
rather than at `t=0`).

The issue asks for SpEC's procedure ("take a few time steps, then update the
control system and map parameters, then rerun") and proposes making it a
repeated phase. Below is what SpEC actually does — which is narrower than
the issue text suggests — and what SpECTRE has.

**Correction to the issue text:** SpEC does **not** re-initialize "all the
time". Its procedure is a *failure-recovery path*, triggered only by four
early-failure reasons within the first ~10 M, capped at 4 attempts
(evidence below). "Doing it all the time as a phase" is a proposal to go
*beyond* SpEC, not to match it — a legitimate choice, but it should be
recorded as such.

## What SpEC does

### Part A — the normal initialization (no iteration)

At `t=0` SpEC seeds the maps directly from the initial-data solve, in
`InputFiles/Bbh/SpatialCoordMap.input:13-31`:

```
InitialData =
FromBeginning
(DtYaw = __Omega__;
 DtExpansion = __aDot__;
 MatchDir = __MapDir__;
 MatchTime = __StartTime__;
 TransCpFile = ID_Init_FuncTrans.txt;
 ExpCpFile = ;  RotCpFile = ;  SkewCpFile = ;
 #initialize shape and size maps from initial data
 ShapeACpFile = ID_Init_FuncLambdaFactorA.txt;
 ShapeBCpFile = ID_Init_FuncLambdaFactorB.txt;
 A0CpFile=ID_Init_FuncLambdaFactorA0.txt;
 B0CpFile=ID_Init_FuncLambdaFactorB0.txt;
 );
```

The key point: **the shape and size maps are initialized from the actual
horizon shape found in the initial-data solve** (`ID_Init_FuncLambdaFactor*`
files produced alongside `ID_AhACoefs.dat`/`ID_AhBCoefs.dat`,
`DoMultipleRuns.input:618-619`), not from an analytic horizon. Consistency
constraint recorded at `DoMultipleRuns.input:382-387` (the `rExc`/`rInitAh`
ratio must match what initial data assumed, "or else shape control will not
have Q=0 at the start of the evolution").

Control system damping timescales are set once from masses/spins
(`DoMultipleRuns.input:271-336`), including the deliberately loose shape
startup (`:288-291`): `TdampingShapeA = 5.0*Tdamping` — *"We initialize the
shape control smoothly in order to avoid wrecking the char speeds."*

### Part B — `RestartWithBetterControlSystemInitialization` (the iteration)

This is SpEC's "take steps, update, rerun", and it is a **failure-recovery
path, not an unconditional startup phase**.

**Trigger.** Only when the run dies in the first ~10 M with one of four
early-failure reasons — `Support/Perl/BatchJobTermination.pm:295-302`:

```perl
} elsif ($reason =~ /^IngoingCharFieldOnSphericalBdry$/ ||
         $reason =~ /^ProportionalIntegral::MaxDt$/ ||
         $reason =~ /^AhC_L\d+ failed\.$/ ||
         $reason =~ /^ProportionalIntegral::MinDt$/) {
  $workdir =
      RestartWithBetterControlSystemInitialization($workdir, $scratch, $InputFiles);
  $change_restart = "FromID";
```

The 10 M window is `TimeThresholdForError`
(`InputFiles/Bbh/Evolution.input:46,50,218`; implementations
`Evolution/FoshSystem/DualFrameSystem/CharSpeedTerminationCriteria.cpp:45,63`
and `Evolution/TimeSteppers/OdeControllerProportionalIntegral.cpp:44,222`).

**Procedure** — `BatchJobTermination.pm:757-823` (inspiral case):

1. Sanity checks: first segment only (`:766-770`), `ID/EvID` must exist
   (`:772-777`).
2. Compute `rminfac = sqrt(ID_rExc{A,B} / ID_r{A,B})` from `ID_Params.perl`
   (`:784-795`).
3. **Rebuild the shape-map initialization from the horizon coefficients the
   failed run actually measured** (`:796-812`), via
   `SurfaceToSpatialCoordMapFiles` on `Ah{A,B}Coefs.dat`, producing new
   `ID_Init_FuncLambdaFactor{A,B}[0].txt`.
4. Hand off to `RestartWithBetterControlSystemInitializationWork`
   (`:491-585`): moves the failed segment aside, counts iterations
   **capping at `max_iter = 4`** (`:496,503-514`), re-points `MatchDir` and
   `SmoothAhRadiusFile` (`:551-582`), and on the last attempt comments out
   every `TimeThresholdForError` so the run cannot loop forever
   (`:532-549`).
5. `$change_restart = "FromID"` (`:303-305`) — the run restarts **from
   `t=0`**, not from where it died, because the map initial values changed.

There is a ringdown variant (`:641-755`) that additionally rebuilds the
translation map from `AhCInertial.dat` and, if the run died within one step,
falls back to `RestartRingdownWithLargerVout` (`:591-639`).

**So the accurate statement of SpEC's behaviour is:** normally, one shot
from ID with no iteration; on early failure, re-derive the shape maps from
the measured horizons and restart from `t=0`, up to 4 times.

## What SpECTRE has today

### Control system state is options-only

`src/ControlSystem/Actions/Initialization.hpp:75-79`:

```cpp
using simple_tags_from_options =
    tmpl::list<control_system::Tags::Averager<ControlSystem>,
               control_system::Tags::Controller<ControlSystem>,
               control_system::Tags::TimescaleTuner<ControlSystem>,
               control_system::Tags::ControlError<ControlSystem>>;
```

Values come from the template (`support/Pipelines/Bbh/Inspiral.yaml:611-718`),
filled by `support/Pipelines/Bbh/Inspiral.py:37-87`
(`_control_system_params`) — a pure function of masses and spins. These
mirror SpEC's `DoMultipleRuns.input:246-336` closely, so the *values* are
largely at parity (term-by-term comparison in the #7418 survey comment).

### The shape map is initialized analytically, not from the measured horizon

`support/Pipelines/Bbh/Inspiral.yaml:180-190` uses
`KerrSchildFromBoyerLindquist`
(`src/Domain/Creators/TimeDependentOptions/ShapeMap.hpp:31-53`) — an
**analytic** Kerr horizon from mass and spin taken from the ID *input* file
(`Inspiral.py:268-275`), not from the ID solve's horizon output.

**The capability to do it SpEC's way already exists and is unused:**
`YlmsFromFile` (`ShapeMap.hpp:64-114`) and `YlmsFromSpEC`
(`ShapeMap.hpp:134-165`), both in the `InitialValues` variant
(`ShapeMap.hpp:272-276`). And the pipeline **already computes the arguments
it would need** — `Inspiral.py:265-267` sets `HorizonsFile`,
`AhASubfileName`, `AhBSubfileName` — which **no template consumes** (grep
over `support/Pipelines/`). Dead plumbing.

This gap is split out as its own concrete implementation issue (see the
cross-referenced issue); what remains here is the re-initialization /
recovery design.

### No re-initialization mechanism

- No SpEC-equivalent recovery path exists. There is no handling of an early
  failure that re-derives map initial values and restarts.
- Control system option tags are **not overlayable**, so they cannot even
  be changed at a checkpoint restart. The complete set of overlayable tags
  (`is_overlayable = true`) is:
  `src/Parallel/PhaseControl/PhaseControlTags.hpp:75`,
  `src/ParallelAlgorithms/EventsAndTriggers/Tags.hpp:93`,
  `src/Time/Tags/LtsStepChoosers.hpp:24`,
  `src/Time/Tags/MinimumTimeStep.hpp:17`,
  `src/Time/Tags/VariableOrderAlgorithm.hpp:20`.
  The overlay machinery: `src/Parallel/ParallelComponentHelpers.hpp:361-401`
  and `src/Parallel/Main.hpp:760-768,846-852,1065-1073`
  (`Parallel::Phase::UpdateOptionsAtRestartFromCheckpoint`,
  `src/Parallel/Phase.hpp:89`).

### The nearest precedent for the proposed "phase" design

`Parallel::Phase::DisableRotationControl` (`src/Parallel/Phase.hpp:55`) plus
`control_system::Actions::SwitchGridRotationToSettle`
(`src/ControlSystem/Actions/GridCenters.hpp:137+`) is an existing phase that
reaches into the control systems mid-run, entered via
`VisitAndReturn<Parallel::Phase::DisableRotationControl>`
(`src/Parallel/PhaseControl/Factory.hpp:16`; used in
`tests/InputFiles/GrMhd/GhValenciaDivClean/BinaryNeutronStar.yaml:49`). It
is a one-way disable, not a re-initialization — but it is the structural
precedent for the issue's "do this as a phase we run `N` times" idea.

## Gap statement

1. Shape map initialized from an analytic Kerr horizon instead of the
   measured ID horizon — **split out as its own issue** (dead plumbing
   exists; see above).
2. SpECTRE has **no recovery path** for the early-failure modes SpEC guards
   against. SpEC's four trigger reasons (`BatchJobTermination.pm:295-302`)
   map onto SpECTRE conditions that currently just kill the run.
3. Control system parameters cannot be changed at restart at all (not
   overlayable), so even a manual "adjust and rerun" requires a fresh run
   with a new input file.
4. Damping timescales and thresholds themselves look close to parity
   (`Inspiral.py:37-87` vs `DoMultipleRuns.input:271-336`) — see the #7418
   survey comment for the term-by-term comparison.

## Prior-art dedupe

| # | kind/state | relevance |
|---|---|---|
| **4254** | **PR merged** | **"Have SpECTRE control system act more like SpECs control system"** — closest prior art; likely records which SpEC behaviours were deliberately not copied. Read first. |
| **3964** | **issue OPEN** | **"Control systems ignore measurements"** — an open control-system defect. Check whether it explains any symptom attributed to initialization before building a re-init phase. |
| 7413 | issue OPEN | The PBJ branch — the same control-system reset at a branch instead of at `t=0` |
| 3590 | PR merged | "Add control system measurement initialization" |
| 3644 | PR merged | "Allow creation of FunctionsOfTime with control system info" |
| 3660 | PR merged | "Construct measurement timescales using control sys info" |
| 4223 | PR merged | "Allow TimescaleTuner to be initialized with arbitrary number of timescales" |
| 4269 | PR merged | "Fix uninitialized timescale tuners from options" |
| 6113 | PR merged | "Add several common time dependent options" — where the shape-map `InitialValues` variant options come from |
| 6952 | PR merged | "Add option to remove delay in FoT update" (2025-11) |

No existing issue or PR covers the shape-map-from-measured-horizon gap —
hence the dedicated issue.

## Proposed design

Sequence the work instead of choosing the big design now. Note SpEC's
procedure updates *map initial values* (shape/size init files), not
controller gains — the control systems themselves restart identically —
so the shape-map initialization is the load-bearing piece:

1. Land the shape-map-from-measured-horizons issue first (split out; the
   dead plumbing makes it cheap). It removes most of SpEC's motivation
   for re-initialization: SpEC's recovery path exists because a wrong
   shape-map init can kill the run in the first ~10 M.
2. Then **measure** early-failure incidence in pipeline runs. Build a
   recovery path only if failures persist.
3. If needed, build it SpEC-shaped and pipeline-level: detect the four
   early-failure terminations, re-derive shape init from the measured
   horizons, resubmit from `t=0` (the evolution done under wrong map
   initial values is invalid anyway), cap attempts (SpEC: 4) — no
   in-executable phase machinery required.
4. Independent enabler, valuable regardless: make control-system tags
   overlayable at checkpoint restart (today they cannot be changed at
   restart at all).

## Open points to settle

- [ ] **OP1 — sequencing**: accept shape-map first, recovery decision
  deferred to measurement? Recommendation: yes — the unconditional
  re-init phase this issue proposed costs a few M of evolution on every
  run; the recovery path costs nothing until something breaks.
- [ ] **OP2 — if a recovery path is built**: pipeline-level resubmit
  (recommendation: matches SpEC, no executable changes) vs an
  in-executable `VisitAndReturn` phase.
- [ ] **OP3 — overlayable control-system tags**: do as independent
  enabling work now, or defer until a concrete use appears?

A follow-up comment settling these points makes this issue ready for
implementation (→ Ready).

---

🤖 drafted by: [Claude Fable 5](https://claude.com/claude-code)
