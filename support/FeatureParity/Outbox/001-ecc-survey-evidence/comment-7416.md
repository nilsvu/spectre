# Survey: ecc reduction iterations at different Levs

Feature-parity survey evidence. Revisions: SpEC @ `5f8f5375ca`, SpECTRE
`develop` @ `4d43624d64` (2026-08-13). All `file:line` references are to
these revisions.

Summary of the verdict: SpECTRE **reuses SpEC's fit and update code
verbatim** (imported at runtime), so the *measurement* is at parity by
construction. Three things around it are not:

- the **two-stage Lev schedule** — stays in this issue (below);
- the **failure/abort conditions** — split out as a dedicated issue (the
  ecc-control loop currently has no iteration cap and no divergence check;
  see the cross-referenced issue);
- the **"smooth continuation" at the end** — the same defect as the PBJ
  branch losing its state, tracked with the PBJ issue #7413 (evidence
  there).

## Mechanism 1 — measuring eccentricity and computing the update

### SpEC

Entry point `Support/DatDataManip/OmegaDotEccRemoval.py` (1585 lines),
driven by `Support/bin/OmegaDotEccRemoval.py`. It fits `dΩ/dt` from the
horizon trajectories.

**Fit window.** `OmegaDotEccRemoval.py:1362-1398`:
- `tmin`: `FindTmin(t, dOmegadt, 500)` (`:42-88`) — running-average of
  `|d²Ω/dt²|` over 10 points, threshold crossing plus a 200 M safety shift,
  **capped at 500 M** (`:84-88`). Purpose: start after junk.
- `tmax`: `min(t[-1], tmin + 5*pi/Omega0)` (`:1393`) — i.e. **2.5 orbits**
  of fitting data.
- `tref` defaults to `tmin` (`:1386-1389`).

**Fit models** (`performNonspinFits` `:618-773`, `performSpinFits`
`:775-980`, variable-projection variants `:982-1353`). Non-spin ladder,
each seeding the next:

| Model | Definition | Line |
|---|---|---|
| `F1` | `a0 (Tc-t)^(-11/8)` | `:638` |
| `F1cos1` | `F1 + B cos(ω t + φ)` | `:647` |
| `F1cos2` | `F1 + B cos(ω t + φ + b t²)` | `:668` |
| `F2cos2` | `a0(Tc-t)^(-11/8) + a1(Tc-t)^(-13/8) + B cos(ω t + φ + b t²)` | `:683-686` |

Spin-spin variants `F1_SS` `:803`, `F1cos1_SS` `:838`, `F1cos2_SS` `:861`,
`F2cos2_SS` — these hold `Tc` fixed from a separate `Ω(t) ∝ (Tc-t)^(-3/8)`
power-law fit (`:792`).

**Update formulas** — `ComputeUpdate` (`:400-502`), citing arXiv:1012.1549
(`:412`):

```python
delta_adot0  = B/(2.*Omega0)*cos(phi)
delta_Omega0 = -B*omega/(4.*Omega0*Omega0)*sin(phi)
if(Improved_Omega0_update):
    delta_Omega0 = -B/(4.*Omega0)*sin(phi)          # extra factor Omega0/omega
delta_D0     = -B*D0*omega*sin(phi)/(2*Omega0*(Omega0**2+2/D0**3))
ecc          = B/(2.*Omega0*omega)
```

(`:415-422`). Only **two of the three** corrections are applied (`:409-412`);
SpEC applies `Omega0` and `adot0` (`Support/Perl/EccReduce.pm:278-279`).

**Eccentricity bound when the fit is poor** (`:429-445`): if `rms/B > 0.4`,
report `ecc = 4*rms/(2*Omega0**2)` as an upper bound.

**Which fit is used** — `Support/Perl/EccReduce.pm:30-47`: `F2cos2_SS`
normally, falling back to the **non**-spin-spin `F2cos2` when `ecc > 0.01`,
because "If the eccentricity is large (>0.01), then the spin-spin terms
should be negligible... they will overfit d(Omega)/dt" (`EccReduce.pm:26-29`).

Invocation for BBH (`EccReduce.pm:245-247`):
```perl
Utils::System("$RealBin/OmegaDotEccRemoval.py -t bbh $xtra_opts".
            "-d=$CombineDir/ApparentHorizons --idperl=$IDFile ".
            "--improved_Omega0_update --tmax=100000000 --no_check");
```
with `--varpro` and `--freq_filter` optionally set from `Params.input`
(`EccReduce.pm:83-88,237-244`).

There are two further SpEC variants, both non-zero-target:
- `Support/bin/InitialDataAdjustment.py` — ecc **control** to a target
  eccentricity using orbital quantities (`EccReduce.pm:254-258`).
- `Support/Python/EOBEccControl.py` (1252 lines) — waveform-based ecc
  control fitting an EOB `A22` to the NR `(2,2)` amplitude.

### SpECTRE

`support/Pipelines/EccentricityControl/EccentricityControlParams.py` calls
**the same SpEC code** (`:92-99`):

```python
check_spec_import(contains_commit="ecfabf1ce78daeacbdd026625a02215c8e84af0e")
from OmegaDotEccRemoval import (ComputeOmegaAndDerivsFromFile, FindTmin, performAllFits)
```

Window (`:128-131`): `tmin = max(FindTmin(t, dOmegadt, 500), t[0])`,
`tmax = min(500 + 5*np.pi/Omega0, t[-1])` — same formulas as SpEC.

Fit call (`:188-215`) uses `opt_freq_filter=True`, `opt_varpro=True`,
`opt_type="bbh"`, `opt_improved_Omega0_update=True`,
`check_periastron_advance=True`.

Trajectories come from `ApparentHorizons/ControlSystemAhA_Centers.dat` /
`..AhB..` (`:42-43`); masses and spins from `ObservationAhA.dat` /
`ObservationAhB.dat` (`:138-184`) with a fallback to `TargetParams`
(`:157-163`).

Hard restriction (`:113-115`): `assert target_eccentricity == 0.0`.

## Mechanism 2 — the iteration loop

### SpEC

Driver `Support/Perl/EccReduce.pm:114-433`, invoked from
`BatchJobTermination.pm:265-277` when the run hits the
`EccentricityReduction` termination criterion
(`Utils/DataBox/EccentricityReduction.cpp:12-53`, wired at
`InputFiles/Bbh/Evolution.input:97-103`).

Directory model: `.../Ecc<N>/{ID,Ev}` (`EccReduce.pm:118-126`). Each
iteration creates `Ecc<N+1>`, copies input files, writes the updated
`Omega0`/`adot0` into `Params.input`, bumps the job name, and submits an
**ID job** (`EccReduce.pm:396-432`).

Run length per iteration — `DoMultipleRuns.input:54,814-819`: **400 M + 2.5
orbits** (`$tmin + 5*pi/$OmegaMeanMotion` with `$tmin = 400`).

**Convergence** (`EccReduce.pm:296-326`):
`abs($Ecc-$TargetEcc) < $EccTolerance`.

**Failure conditions** — both present, both absent in SpECTRE (split out as
a dedicated issue):
1. *Converging too slowly* (`EccReduce.pm:335-371`): if
   `abs(Ecc-TargetEcc) > abs(OldEcc-OldEccT)` **and** the previous iteration
   used the same Lev, abort with `IsError=1` (`:367`). The same-Lev guard at
   `:350-351` exists precisely so that the rough→final Lev switch is not
   mistaken for divergence.
2. *Too many iterations* (`EccReduce.pm:373-388`): `MaxIts` default 7
   (`EccentricityReduction.cpp:18`).

### SpECTRE

`support/Pipelines/Bbh/EccentricityControl.py:27-176`, re-entered through
the `Next:` block of the input file
(`support/Pipelines/Bbh/Inspiral.yaml:19-37`).

Run length per iteration — `support/Pipelines/Bbh/Inspiral.py:504-509`:
`500 + 5π/Ω0` (SpEC uses 400, SpECTRE 500, otherwise identical — the
difference is unexplained). Enforced by a plain `TimeCompares` trigger
(`Inspiral.yaml:472-475`).

Convergence (`EccentricityControl.py:115-118`): absolute tolerance, default
`1e-3` (`support/Pipelines/Bbh/InitialData.py:41-43`).

Otherwise it regenerates initial data (`EccentricityControl.py:143-176`)
with `orbital_angular_velocity=ecc_params["NewOmega0"]`,
`radial_expansion_velocity=ecc_params["NewAdot0"]`, and
`eccentricity_control=True` so the loop repeats — with no iteration cap and
no divergence check (the dedicated issue).

## Gap kept in this issue — no two-stage Lev schedule (rough → final)

SpEC does the first iterations cheaply at a low Lev and only the last ones
at the production Lev. `DoMultipleRuns.input:30-46`:

```perl
my $EccRedLev = $MaxLev;
my $TargetEcc = 7e-4;
# The first few iterations of eccentricity reduction are done using
# $InitialEccRedLev until the eccentricity goes below $InitialTargetEcc. We
# refer to this as RoughEccReduction; this is much faster than using $MaxLev
# but is only approximate.
my $InitialEccRedLev = $MinLev;
my $InitialTargetEcc = 1e-3;
my $RoughEccReduction = ($InitialEccRedLev == $EccRedLev) ? 0 : 1;
```

The switch from rough to final is a rewrite of the child
`DoMultipleRuns.input` (`EccReduce.pm:327-334`), and it also happens on the
two failure paths (`EccReduce.pm:357-363,377-383`) — i.e. "if rough
reduction stalls, promote to the final Lev rather than aborting".

**SpECTRE has none of this.** The Lev is fixed for the whole ecc-control
loop: `TargetParams["EvolutionLev"]`
(`support/Pipelines/Bbh/InitialData.py:38`) is passed once at
`PostprocessId.py:186`, and there is a single
`EccentricityAbsoluteTolerance`. The numbers to port are `1e-3` (rough) →
`7e-4` (final), `MinLev` → `MaxLev`.

Note: issue #6460 ("Ecc control: run at lower resolution for first few
iterations", empty body) describes this same gap and is proposed to be
closed as superseded by this issue.

## Option drift on the shared fit — question, not defect

Both codes call `performAllFits`, but with different options: SpEC
production passes `--no_check` (`check_periastron_advance=False`) and
enables varpro/freq-filter only on request (`EccReduce.pm:237-247`);
SpECTRE hard-codes `opt_varpro=True`, `opt_freq_filter=True`,
`check_periastron_advance=True` (`EccentricityControlParams.py:188-215`).
Also SpEC selects between `Params_F2cos2_SS.dat` and `Params_F2cos2.dat`
based on `ecc > 0.01` (`EccReduce.pm:34-39`); SpECTRE takes whatever
`performAllFits` returns. If these are intentional improvements they should
be recorded — especially before the fit code moves to SimulationSupport
(#7412), where both codes will share it; if not, they are a silent
divergence.

## Nonzero target eccentricity — out of scope, recorded

SpEC supports three target regimes (reduction to zero; control to a target
via `InitialDataAdjustment.py`; EOB waveform-based control via
`EOBEccControl.py`). SpECTRE asserts `eccentricity == 0.0` in two places
(`EccentricityControlParams.py:113-115`,
`InitialOrbitalParameters.py:85-88`). Out of scope for this issue as
written, but it is the reason `TargetParams` already carries
`MeanAnomalyFraction` (`InitialData.py:34`) that nothing reads.

## Prior-art dedupe

| # | kind/state | relevance |
|---|---|---|
| **6460** | **issue OPEN** | **Duplicate of the two-stage Lev schedule gap** (rough ecc reduction at low Lev) |
| **6944** | **PR merged** | **"EccControl: fix continuation after ecc-control is complete"** — the current continuation behaviour; read before redesigning |
| 6406 | PR merged | "Eccentricity Reduction Automation" — the automation loop this issue extends |
| 6295 | PR merged | "Single iteration of eccentricity control reduction" |
| 6333 | PR merged | "Use SpEC's eccentricity control" — created the runtime SpEC dependency |
| 6467 | PR merged | "Check SpEC version when importing" — the `check_spec_import` guard |
| 6490 | PR merged | "BBH pipeline: set time bounds in ecc control" — the `tmin`/`tmax` window |
| 6468 | PR merged | "BBH pipeline: write eccentricity params to file, add CLI to compute params & plot" |
| 6578 / 6465 / 6461 / 6290 / 6190 / 6927 | PR merged | Assorted ecc-control fixes (file ordering, glob patterns, subfile names, output, CLI defaults) |
| **5966** | **PR closed** | **"Add OmegaDotEccentricityControl.py for SpECTRE"** (2024-08) — an abandoned attempt to port SpEC's `OmegaDotEccRemoval` into SpECTRE. Direct prior art for the SimulationSupport port; read why it was dropped first. |
| 5895 | PR merged | "Add script to compute eccentricity removal updates for SpECTRE ID" — the original update script |
| 5909 | issue CLOSED | "Automate eccentricity control" — closed 2025-04-07 |
| 5892 | issue OPEN | "Add CLI that runs SpEC ecc reduction script" — frames the SpEC script as temporary |
| 5937 | issue OPEN | "Initial orbital parameters and eccentricity control" — broader older tracker |
| 7414 | issue OPEN | "We will need to add support for EccControl." Empty body, not on the project board; overlaps this cluster |
| 6890 | PR draft | "Ecc control: depend on SimulationSupport instead of SpEC" — targets the `from OmegaDotEccRemoval import ...` line. **Its `OmegaDotEccRemoval` import is broken: SimulationSupport has no such module.** Full analysis in the survey comment on #7412. |
| 7089 | issue OPEN | "Nondeterministic pipeline test failures" — touches ecc-control pipeline tests |

**Nothing existing covers the abort conditions.** That is the one item from
this issue's original scope with no prior art at all, and the one with the
clearest cost consequence — hence the dedicated issue.

---

drafted by: Claude Fable 5
