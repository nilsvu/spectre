# Survey: control system initialization at run start

Feature-parity survey evidence. Revisions: SpEC @ sxs-collaboration/spec@5f8f5375ca, SpECTRE `develop` @ sxs-collaboration/spectre@4d43624d64 (2026-08-13). All `file:line` references are to these revisions. Companion: #7413 (the same state problem at a PBJ branch rather than at `t=0`).

**Correction to the issue text:** SpEC does **not** re-initialize "all the time". Its procedure is a *failure-recovery path*, triggered only by four early-failure reasons within the first ~10 M, capped at 4 attempts (evidence below). "Doing it all the time as a phase" is a proposal to go *beyond* SpEC, not to match it — a legitimate choice, but it should be recorded as such.

## What SpEC does

**Normal initialization — one shot, no iteration:**

- Shape/size maps are seeded from the **measured** ID horizon (`ShapeACpFile = ID_Init_FuncLambdaFactorA.txt` etc., `InputFiles/Bbh/SpatialCoordMap.input:26-30`).
- A consistency constraint is recorded: the `rExc`/`rInitAh` ratio must match what initial data assumed, "or else shape control will not have Q=0 at the start of the evolution" (`DoMultipleRuns.input:382-387`).
- Damping timescales are set once from masses/spins (`DoMultipleRuns.input:271-336`); the shape startup is deliberately loose (`TdampingShapeA = 5.0*Tdamping`, *"to avoid wrecking the char speeds"*, `:288-291`).

**`RestartWithBetterControlSystemInitialization` — the iteration, on failure only:**

- Fires only when the run dies within `TimeThresholdForError ≈ 10` M with one of four reasons: ingoing char fields at the excision boundary, ODE-controller Max/MinDt, or AhC failure (`Support/Perl/BatchJobTermination.pm:295-302`).
- It **rebuilds the shape-map init files from the horizon coefficients the failed run actually measured** (`SurfaceToSpatialCoordMapFiles` on `Ah{A,B}Coefs.dat`, `:796-812`).
- Attempts are capped at 4 (`:496`); the last attempt disables the error thresholds so the run cannot loop (`:532-549`).
- The restart is **from `t=0`** (`$change_restart = "FromID"`) — the map initial values changed, invalidating the evolution already done.
- A ringdown variant also rebuilds the translation map (`:641-755`).

## What SpECTRE has today

- **Control-system state is options-only**: Averager, Controller, TimescaleTuner, ControlError are `simple_tags_from_options` (`src/ControlSystem/Actions/Initialization.hpp:75-79`), filled from template values computed from masses/spins (`Inspiral.py:37-87`). The *values* closely mirror SpEC's — term-by-term comparison in the #7418 survey comment.
- **The shape map is initialized analytically** (`KerrSchildFromBoyerLindquist` from the ID *input* file, `Inspiral.yaml:180-190`), not from the ID solve's horizon output. The SpEC-equivalent options exist unused (`YlmsFromFile` / `YlmsFromSpEC`, `ShapeMap.hpp:64-165`), and `Inspiral.py:265-267` already computes `HorizonsFile`/`Ah{A,B}SubfileName` — **which no template consumes**. Dead plumbing; split out as its own implementation issue (cross-referenced).
- **No recovery path** exists for the early-failure modes SpEC guards against — they currently just kill the run.
- **Control-system tags are not overlayable** at checkpoint restart (machinery: `Parallel::Phase::UpdateOptionsAtRestartFromCheckpoint`, `src/Parallel/ParallelComponentHelpers.hpp:361-401`) — even a manual "adjust and rerun" requires a fresh run.
- Nearest precedent for an in-executable phase design: `Parallel::Phase::DisableRotationControl` + `SwitchGridRotationToSettle` (`src/ControlSystem/Actions/GridCenters.hpp:137+`) — a one-way disable, not a re-init, but the structural pattern.

## Prior art

| # | state | relevance |
|---|---|---|
| **#4254** | PR merged | "Have SpECTRE control system act more like SpECs control system" — closest prior art; likely records which SpEC behaviours were deliberately not copied. Read first. |
| **#3964** | issue OPEN | "Control systems ignore measurements" — open defect; rule it out before attributing symptoms to initialization. |
| #7413 | issue OPEN | the same control-system reset at the PBJ branch |
| #3590 / #3644 / #3660 / #4223 / #4269 / #6113 / #6952 | PR merged | control-system initialization and shape-map options history |

No existing issue or PR covers the shape-map-from-measured-horizon gap — hence the dedicated issue.

## Proposed design

Sequence the work instead of choosing the big design now. Note SpEC's procedure updates *map initial values* (shape/size init files), not controller gains — the control systems themselves restart identically — so the shape-map initialization is the load-bearing piece:

1. Land the shape-map-from-measured-horizons issue first (split out; the dead plumbing makes it cheap). It removes most of SpEC's motivation for re-initialization: SpEC's recovery path exists because a wrong shape-map init can kill the run in the first ~10 M.
2. Then **measure** early-failure incidence in pipeline runs. Build a recovery path only if failures persist.
3. If needed, build it SpEC-shaped and pipeline-level: the pipeline detects the early-failure terminations (the SpECTRE analogues of SpEC's four trigger reasons — char-speed violation at the excision boundary, step-size floor (`MinimumTimeStep`), horizon-finder failure — within a `TimeThresholdForError`-like window of ~10 M), re-derives the shape-map initial values from the horizons the failed run measured (the `YlmsFromFile` path of the split issue, pointed at the failed run's horizon output), resubmits from `t=0`, and caps attempts (SpEC: 4). No in-executable phase machinery required.
4. Independent enabler, valuable regardless: make the control-system tags overlayable at checkpoint restart (`Averager`/`Controller`/`TimescaleTuner`/`ControlError`, `Initialization.hpp:75-79` — today none of them can be changed at restart).

**Testing / acceptance** (for step 3, if built): a pipeline test that injects an early failure and observes one re-derived resubmission and the attempt cap; the recovery path never triggers on runs that pass the first ~10 M.

## Open points to settle

- [ ] **1. Sequencing** — accept shape-map first, recovery decision deferred to measurement? Recommendation: yes — the unconditional re-init phase this issue proposed costs a few M of evolution on every run; the recovery path costs nothing until something breaks.
- [ ] **2. Recovery mechanism** (if built) — pipeline-level resubmit (recommendation: matches SpEC, no executable changes) vs an in-executable `VisitAndReturn` phase.
- [ ] **3. Overlayable control-system tags** — do as independent enabling work now, or defer until a concrete use appears?

A follow-up comment settling these points makes this issue ready for implementation (→ Ready).

---

🤖 drafted by: [Claude Fable 5](https://claude.com/claude-code)
(and reviewed by me 🙋)
