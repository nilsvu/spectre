# Survey: the PBJ branch — multiple Levs, and the state carried across the branch

Feature-parity survey evidence. Revisions: SpEC @ `5f8f5375ca`, SpECTRE
`develop` @ `4d43624d64` (2026-08-13). All `file:line` references are to
these revisions.

This comment covers both halves of the PBJ topic — the *workflow* of
branching to multiple Levs and adding one later (this issue), and the
*state* carried across the branch (time stepper + control systems, filed as
#7411; the two are one code path and are proposed to be tracked together
here).

**Headline: SpECTRE already has a PBJ branch mechanism (shipped by PR #6840
and #6445). Three things are wrong with it: it branches to exactly one Lev,
there is no path to add a Lev later, and the state carried across the branch
is discarded.**

## What SpEC does — PBandJ ("Perform Branching after Junk")

SpEC runs a single base Lev through junk and (if requested) through
eccentricity reduction, then **branches all other Levs off that one run** so
every Lev shares identical post-junk data. Implementation:
`Support/Perl/PBandJ.pm` (315 lines).

### Configuration

`InputFiles/Bbh/DoMultipleRuns.input:15-17,59-64`:

```perl
my $MinLev = 1;
my $MaxLev = 3;
...
# Is this a PBandJ (Perform Branching after Junk) run? (1 for yes, 0 for no).
# If PBandJ=1, the branching into multiple Levs is done after a certain time
# so that all Levs have exactly the same post-junk initial data.
# For EccRedRun = 1 this is done at the time when the eccentricity has been
# deemed acceptibly small.
my $PBandJ = 1;
```

Branch time `DoMultipleRuns.input:763-765`:

```perl
my $ActivatePBandJTimeTermination = $PBandJ;
my $PBandJTime = 1.2 * $Rmax;
my $PBandJBaseLev = $MaxLev;
```

i.e. **1.2 outer-boundary light-crossing times**, after the junk has left
the domain (termination criterion `PBandJTime` —
`Utils/DataBox/PBandJTime.cpp:12-38`, wired in
`InputFiles/Bbh/Evolution.input:104-107`). Only the base Lev is generated at
`t=0` (`DoMultipleRuns.input:821`):
`next if($PBandJ and not $k == $PBandJBaseLev);`

For eccentricity-reduction runs the `PBandJTime` criterion is disabled and
the branch is taken at the ecc-reduction termination instead
(`DoMultipleRuns.input:767-771`) — this matches SpECTRE's
`branch_levs_when_complete` placement.

### The three-phase mechanism

**Phase 1 — `StartOtherLevs`** (`PBandJ.pm:107-207`), called at the branch
from `BatchJobTermination.pm:278-282` (`PBandJTime`) or from
`EccReduce.pm:319` (ecc-reduction convergence):

1. Turns off `PBandJ`, `EccRedRun`, `EOBEccControl` in the child
   `DoMultipleRuns.input` (`PBandJ.pm:114-118`).
2. Runs `DoMultipleRuns -n -s '^Lev<base>_[A-Z]+$'` (`PBandJ.pm:152,166`)
   — `-n` = create directories but do not submit
   (`Support/DoMultipleRuns.pl:135`), `-s REGEXP` = skip existing dirs
   (`DoMultipleRuns.pl:148-153`). This generates `Lev*_AA` for the *other*
   Levs purely to obtain their `AmrTolerances.input`.
3. `MakeLinksForPBandJ` (`PBandJ.pm:51-96`) renames each other Lev's
   `Lev*_AA` to `Lev*_AA_PBandJBackup` (`:78`) and symlinks **every base-Lev
   segment up to the branch segment** into the other Levs
   (`:83-93`), so pre-branch history is literally shared, not duplicated.
4. Writes `ProhibitStartJobReruns.txt` (`PBandJ.pm:191-205`) to stop a user
   from starting the other Levs from `t=0`.

**Phase 2 — `SubmitOtherLevsIfPBandJ`** (`PBandJ.pm:226-314`), called from
`BatchJobTermination.pm:375-382` after the new base-Lev segment exists. For
each non-base Lev:

- Creates `Lev<N>_<suffix>` and links the SpEC binaries (`:254-255`).
- Substitutes that Lev's `AmrTolerances.input` from the backup dir
  (`:258-259`).
- Rewrites the restart to interpolate onto the new resolution
  (`:262-272`):
  ```perl
  SpEC::ChangeEvolutionInputToRestartFromLastStep($scratch, 1, $InputFiles, "Evolution.input");
  my $to_add = "FlagAllSubdomainsAsChanged = yes; " .
               "ResetFilterFunctionsForAMR = yes; " .
               "StartAmrWithMinimumExtents = yes; ";
  ```
  `Interp = 1` produces
  `GlobalVarsCheckpoint = Interpolated(ResolutionChanger=Spectral; ...)` —
  see `Support/Perl/SpEC.pm:186-192`. **This is the actual resolution
  change: a spectral-interpolating checkpoint restart.**
- Rewrites `Level = <base>;` → `Level = <N>;` in `GrDomain.input` and
  `AmrDriver.input` (`:283-288`).
- Deletes the backup dir, renames the job, and submits (`:292-310`).

### Adding a Lev *later*, from an already-completed run

This is the exact scenario in the issue title, and SpEC supports it
explicitly and idempotently:

- `StartOtherLevs` detects Levs already branched and excludes them
  (`PBandJ.pm:129-151`); `MakeLinksForPBandJ` skips already-linked Levs
  (`PBandJ.pm:73-77`).
- The user-facing recipe is written into the guard file
  (`PBandJ.pm:196-202`):
  > To run a new Lev, do the following instead:
  > Run MakeNextSegment from the Lev`$BaseLev` segment that terminates with
  > termination condition = EccentricityReduction or condition = PBandJTime.

So: bump `$MaxLev` in `DoMultipleRuns.input`, then re-run `MakeNextSegment`
on the branch segment. The pre-branch segments get symlinked, the branch
segment gets copied and interpolated to the new Lev.

End-to-end coverage: `Support/Tests/TestPbAndJ/run_test` and
`Support/Tests/TestPbAndJWithoutEccRed`.

## What SpECTRE has today

**A PBJ branch exists.** `support/Pipelines/Bbh/EccentricityControl.py:120-140`:

```python
if branch_levs_when_complete:
    for lev in branch_levs_when_complete:
        ...
        start_inspiral(
            id_input_file_path=inspiral_input_file_path,
            id_run_dir=inspiral_run_dir,
            id_subfile_name="PostJunkVolumeData",
            lev=lev, continue_with_ringdown=True,
            pipeline_dir=lev_dir.path, **scheduler_kwargs)
```

The branch data: `support/Pipelines/Bbh/Inspiral.yaml:470-489` writes
`SpacetimeMetric, Pi, Phi` to subfile `PostJunkVolumeData` at
`TimeCompares >= FinalTime`, then `Completion`.

Re-import: `Inspiral.py:165` (`id_from_evolution = "Evolution" in
id_input_file`), `Inspiral.py:240-250` (reads `InitialTime` and
`FotFilename` from the volume file), `Inspiral.yaml:50-64`
(`NumericInitialData` with `ObservationValue: {{ InitialTime }}`,
`ElementsAreIdentical: True`), `Inspiral.yaml:143-170` (every
time-dependent map read `FromVolumeFile`).

**So: fields and maps continue correctly.** Levs are pure p-refinement —
`Inspiral.py:24-33`: `refinement_level: 1`, `polynomial_order: 7 + lev`,
comment *"To be replaced once AMR is used."*

## Gaps

### G1 — the automated pipeline branches to exactly one Lev

`support/Pipelines/Bbh/Inspiral.yaml:29`:

```yaml
branch_levs_when_complete: [ {{ Lev }} ]
```

The Python function accepts a list, but the template hard-wires a
single-element list containing the current Lev. SpEC's default
(`MinLev=1, MaxLev=3`) branches **three** Levs. This is a one-line-shaped
gap in the template plus a decision about where the Lev list comes from.

### G2 — no path to add a Lev after the fact

SpEC has an explicit, idempotent re-entry (above). SpECTRE has no CLI verb
for "add Lev N to this completed pipeline"; one would have to hand-call
`start_inspiral` with the right five arguments. Nothing prevents it, nothing
supports it, and nothing records which Levs already branched.

### G3 — the branch discards the state that makes it a continuation

Three independent defects, previously filed as #7411:

1. **Time step.** `InitialTimeStep: 0.0002` and `InitialSlabSize: 0.1` are
   **hard-coded literals, not Jinja-templated** (`Inspiral.yaml:220,225`) —
   unchanged on the `IdFromEvolution` branch. They are values chosen for
   `t = 0` with junk present; a PBJ branch happens at a settled time, where
   the natural step is orders of magnitude larger.
2. **Multistep history.** `AdamsMoultonPcMonotonic` order 4
   (`Inspiral.yaml:242-244`) is a multistep predictor–corrector
   (`src/Time/TimeSteppers/AdamsMoultonPc.hpp:50-51`, `minimum_order = 2` at
   `:104`). A branch from imported volume data starts with empty history, so
   it restarts at minimum order and re-ramps, on top of the too-small step.
3. **Control-system state.** The maps are continued, but the controllers
   that drive them are not. Averager, Controller, TimescaleTuner and
   ControlError are all `simple_tags_from_options`
   (`src/ControlSystem/Actions/Initialization.hpp:75-79`, `apply` at
   `:107-126` zeroes the measurement counter) — read from literal template
   values (`Inspiral.yaml:611-718`), which are a pure function of masses and
   spins (`Inspiral.py:37-87`), identical at `t=0` and at the branch time.
   At the branch the maps move at their settled rates but the controllers
   believe they are at `t=0` with the deliberately loose startup timescales
   (compare SpEC's `TdampingShapeA = 5.0*Tdamping`, *"We initialize the
   shape control smoothly in order to avoid wrecking the char speeds"* —
   `DoMultipleRuns.input:288-291`) and no measurement history.

**Contrast with SpEC.** SpEC's branch is a `FromLastStep` checkpoint restart
(`PBandJ.pm:262-263` → `SpEC::ChangeEvolutionInputToRestartFromLastStep`,
`Support/Perl/SpEC.pm:168-257`) with a single-step integrator
(DormandPrince5 + `AdaptiveDense`, `InputFiles/Bbh/Evolution.input:213-220`)
— the step size is restored from the checkpoint, there is no multistep
history to lose, and control-system state comes back with everything else.
The only things SpEC deliberately resets at a branch are AMR-related
(`PBandJ.pm:264-271`). **SpEC has no equivalent problem and offers no recipe
to copy — this is a SpECTRE-specific design question.**

### G4 — `ElementsAreIdentical: True` on the branch

`Inspiral.yaml:57`. Currently *correct* only because `refinement_level` is
fixed at 1 for every Lev and only `polynomial_order` changes, so the element
decomposition is identical across Levs. Two consequences:

- The interpolation is p-only, done by the importer; there is no
  h-refinement analogue of SpEC's `ResolutionChanger=Spectral` +
  `FlagAllSubdomainsAsChanged` + `StartAmrWithMinimumExtents`
  (`PBandJ.pm:262-271`).
- If Levs ever differ in `refinement_level` (or AMR is turned on —
  `Inspiral.yaml:313-324` currently has an **empty `Criteria:`**, so AMR is
  configured but inert), this flag becomes wrong. Worth a guard.

### G5 — no shared pre-branch history

SpEC symlinks the base-Lev segments into every other Lev (`PBandJ.pm:83-93`)
so the shared history is visible and not duplicated. SpECTRE's
`PipelineStep`/`Segment` layout (`support/Python/DirectoryStructure.py:132-193`)
puts each branched Lev in its own subdirectory with no back-reference to the
shared parent segments. Cosmetic for correctness, real for post-processing
and archiving (SpEC's catalog tooling relies on all Levs having identical
pre-branch data at identical paths).

## Prior-art dedupe

| # | kind/state | relevance |
|---|---|---|
| 6840 | PR merged | "Ecc control: branch levs when complete (PBJ)" — **shipped the mechanism**; this issue is about extending it |
| 6445 | PR merged | "BBH pipeline: store target params, continue from evolution data for PBJ" — the continuation path |
| **6717** | **PR open/draft** | **"Add branch runs command to pipeline"** — a generic run-branching CLI. Plausibly the right home for an "add a Lev later" verb. **Check before writing a new command.** |
| **6849** | **issue OPEN** | **"Enable full checkpoint/restart without charm++ checkpoints & PBJ restarts"** — directly addresses the "should the branch be a checkpoint restart" question. **Read before designing.** |
| 6953 | PR merged | "Add ringdown levs" — Lev handling for the ringdown stage |
| 6952 | PR merged | "Add option to remove delay in FoT update" (2025-11) — recent FoT update timing work |
| 3964 | issue OPEN | "Control systems ignore measurements" — an independent open control-system defect; check it is not the same symptom before attributing behaviour to the branch |
| 6460 | issue OPEN | "Ecc control: run at lower resolution for first few iterations" — the Lev *schedule* during ecc control, not the branch; belongs with #7416 |

No existing issue or PR addresses the time-stepper half of G3.

## Open design questions

- **Does the branch stay a volume-data import, or become a checkpoint
  restart?** A checkpoint restart fixes G3 outright but cannot change
  resolution — which is the entire point of this issue. If Levs must differ,
  the import path is forced and G3 must be fixed *within* it. #6849 first.
- If it stays an import: which state gets serialized into the branch data?
  Minimum viable set appears to be `{last time step, control-system
  timescales, averager history}`.
- Alternatively: accept a short re-settling window after the branch and
  *measure* that the waveform is unaffected. Cheaper, and testable.
- Where does the Lev list come from? Candidates: a new `TargetParams` field
  (there is already `EvolutionLev` — `support/Pipelines/Bbh/InitialData.py:38`
  used at `PostprocessId.py:186`), a CLI option on `generate-id`, or a
  `MinLev`/`MaxLev` pair mirroring SpEC.
- Should adding a Lev later be a first-class CLI verb (e.g.
  `spectre bbh add-lev -d <pipeline_dir> --lev 4`), recording branched Levs
  so it is idempotent the way SpEC's symlink check is?

---

🤖 drafted by: [Claude Fable 5](https://claude.com/claude-code)
