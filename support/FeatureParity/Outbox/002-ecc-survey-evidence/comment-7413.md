# Survey: the PBJ branch — multiple Levs, and the state carried across the branch

Feature-parity survey evidence. Revisions: SpEC @ `5f8f5375ca`, SpECTRE `develop` @ `4d43624d64` (2026-08-13). All `file:line` references are to these revisions. Covers both halves of the PBJ topic — the *workflow* (multiple Levs, adding one later; this issue) and the *state* carried across the branch (time stepper + control systems; filed as #7411, one code path, tracked together here).

**Headline: SpECTRE already has a PBJ branch mechanism (shipped by PR
#6840 and #6445). Three things are wrong with it: it branches to exactly
one Lev, there is no path to add a Lev later, and the state carried across the branch is discarded.**

## What SpEC does — PBandJ ("Perform Branching after Junk")

SpEC runs a single base Lev through junk and (if requested) through eccentricity reduction, then branches all other Levs off that one run, so every Lev shares identical post-junk data (`Support/Perl/PBandJ.pm`; `MinLev=1`/`MaxLev=3`, branch at `PBandJTime = 1.2·Rmax` or at ecc-reduction convergence — `DoMultipleRuns.input:15-17,59-64,763-771`). What matters for the port:

- **Pre-branch history is symlinked** into the other Levs, not duplicated (`PBandJ.pm:83-93`).
- **The resolution change is a spectral-interpolating checkpoint restart**: `ChangeEvolutionInputToRestartFromLastStep` with `Interp=1`, plus AMR reset flags (`PBandJ.pm:262-288`, `Support/Perl/SpEC.pm:186-192`).
- **Adding a Lev later is explicit and idempotent**: already-branched Levs are detected and skipped (`PBandJ.pm:73-77,129-151`); the user recipe (bump `$MaxLev`, re-run `MakeNextSegment` on the branch segment) is written into a guard file (`PBandJ.pm:196-202`). End-to-end tests exist (`Support/Tests/TestPbAndJ*`).

## What SpECTRE has today

- The branch exists: `EccentricityControl.py:120-140` calls `start_inspiral(..., id_subfile_name="PostJunkVolumeData", lev=...)` per requested Lev; `Inspiral.yaml:470-489` writes the branch data.
- The re-import carries the fields and **all** time-dependent maps (`Inspiral.py:165,240-250`; `Inspiral.yaml:50-64,143-170`, `FromVolumeFile`, `ElementsAreIdentical: True`). **Fields and maps continue correctly.**
- Levs are pure p-refinement: `refinement_level: 1`, `polynomial_order: 7 + lev` (`Inspiral.py:24-33`).

## Gaps

**G1 — the pipeline branches to exactly one Lev.** `Inspiral.yaml:29`:

```yaml
branch_levs_when_complete: [ {{ Lev }} ]
```

The Python accepts a list; the template hard-wires one element — the Lev already running. SpEC's default branches three.

**G2 — no path to add a Lev after the fact.** No CLI verb, and nothing records which Levs already branched (SpEC's re-entry is idempotent, above).

**G3 — the branch discards the state that makes it a continuation** (previously #7411). Three independent defects:

1. *Time step*: `InitialTimeStep: 0.0002` and `InitialSlabSize: 0.1` are hard-coded literals (`Inspiral.yaml:220,225`) chosen for `t=0` with junk present — wrong at a settled branch time, where the natural step is orders of magnitude larger.
2. *Multistep history*: `AdamsMoultonPcMonotonic` order 4 starts the branch with empty history, so it restarts at `minimum_order = 2` (`src/Time/TimeSteppers/AdamsMoultonPc.hpp:104`) and re-ramps — on top of the too-small step.
3. *Control-system state*: the maps continue but their controllers do not. Averager, Controller, TimescaleTuner and ControlError are all `simple_tags_from_options` (`src/ControlSystem/Actions/Initialization.hpp:75-79`), filled from template values that are a pure function of masses/spins (`Inspiral.py:37-87`) — identical at `t=0` and at the branch. So the controllers restart with the deliberately loose startup timescales and zero measurement history while the maps move at settled rates.

*Contrast*: SpEC's branch is a `FromLastStep` checkpoint restart with a single-step integrator (DormandPrince5), so step size and control-system state come back from the checkpoint. **There is no SpEC recipe to port — this is a SpECTRE-specific design question.**

**G4 — `ElementsAreIdentical: True` is only accidentally correct** (`Inspiral.yaml:57`): valid while every Lev has `refinement_level: 1`. If Levs ever differ in h-refinement (or AMR is enabled — the `Amr:` block at `Inspiral.yaml:313-324` has an empty `Criteria:` today), the flag becomes wrong. Worth a guard.

**G5 — no shared pre-branch history.** Each branched Lev gets its own `PipelineStep` subdirectory with no back-reference to the shared parent segments (`support/Python/DirectoryStructure.py:132-193`); SpEC symlinks. Cosmetic for correctness, real for post-processing and archiving.

## Prior art

| # | state | relevance |
|---|---|---|
| 6840 / 6445 | PR merged | shipped the branch mechanism and continuation path — this issue extends them |
| **6717** | **PR open/draft** | **"Add branch runs command to pipeline"** — plausibly the right home for an add-a-Lev verb. Check before writing a new command. |
| **6849** | **issue OPEN** | **"Enable full checkpoint/restart without charm++ checkpoints & PBJ restarts"** — bears on import-vs-checkpoint. Read before designing. |
| 3964 | issue OPEN | "Control systems ignore measurements" — independent defect; rule it out before attributing symptoms to the branch |

No existing issue or PR addresses the time-stepper half of G3.

## Proposed design

Keep the branch a **volume-data import** — a checkpoint restart cannot change resolution, and changing resolution is the point of this issue (#6849 tracks checkpoint-restart work; it complements rather than replaces this). All changes are pipeline-side (Python + template); no executable changes.

1. **Time step across the branch.** Template `InitialTimeStep` and `InitialSlabSize` (today literals, `Inspiral.yaml:220,225`) with the current values as defaults. On the `IdFromEvolution` branch (`Inspiral.py:240-250`, where `InitialTime` and `FotFilename` are already read from the parent), also read the parent's final time step and slab size — from the recorded time-step data in the parent's reductions file if present, else record them alongside `PostJunkVolumeData` at the branch observation — and fill the template with them. Accept the multistep order re-ramp (a few steps at reduced order at a settled time); quantify it once in validation.
2. **Control-system state across the branch.** The tuned damping timescales are already written to disk by the control systems (`WriteDataToDisk: true`, `Inspiral.yaml:612`). On the branch, read each system's final damping timescale from the parent's reductions file and fill `InitialTimescales` (`Inspiral.yaml:611-718`) with the measured values instead of the mass/spin formulas — the same read-from-parent pattern the maps already use (`FromVolumeFile`). Averager history is deliberately **not** carried initially: it is not on disk, and the expected cost is a short re-settling — measured in validation (open point 2).
3. **Lev list.** `TargetParams` gains `BranchLevs` (list of ints, default `[EvolutionLev]`), threaded to `branch_levs_when_complete` in the template, replacing the hard-wired `[ {{ Lev }} ]`.
4. **Add a Lev later.** CLI verb `spectre bbh add-lev -d <pipeline_dir> --lev N`: locates the branch-point segment, calls `start_inspiral` with the recorded arguments, refuses Levs already branched — i.e. idempotent like SpEC's symlink check. Check PR #6717 (generic branch-runs command) as the home before adding a new verb.
5. **Guard.** Error out if `ElementsAreIdentical: True` would be used with a `refinement_level` differing between parent and branch Lev.

**Testing / acceptance**: unit tests for the template rendering (branch vs `t=0` paths yield the intended time-step and timescale values) and for `add-lev` idempotence; the ecc-control pipeline test extended to branch two Levs; one validation run comparing a branched evolution against an uninterrupted continuation at the same Lev — acceptance is time-step recovery within a few steps, control-system timescales continuous at the branch, and waveform differences confined to the re-settling window.

## Open points to settle

- [ ] **1. Import vs checkpoint** — confirm the import path. Recommendation: yes; resolution change requires it. Coordinate with #6849 rather than wait for it.
- [ ] **2. State carried** — tuned timescales + last time step (recommendation), also averager history (requires serializing state that is not on disk today), or time step only with full control re-settling. The validation run adjudicates whether the accepted loss is visible in the waveform.
- [ ] **3. Lev list shape** — `BranchLevs` list in `TargetParams` (recommendation) vs a `MinLev`/`MaxLev` pair mirroring SpEC.
- [ ] **4. add-lev home** — extend PR #6717's branch-runs command vs a new `spectre bbh add-lev` verb.

A follow-up comment settling these points makes this issue ready for implementation (→ Ready).

---

🤖 drafted by: [Claude Fable 5](https://claude.com/claude-code)
(and reviewed by me 🙋)
