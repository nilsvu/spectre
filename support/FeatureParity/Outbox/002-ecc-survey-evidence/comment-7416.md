# Survey: ecc reduction iterations at different Levs

Feature-parity survey evidence. Revisions: SpEC @ `5f8f5375ca`, SpECTRE `develop` @ `4d43624d64` (2026-08-13). All `file:line` references are to these revisions.

**Verdict: SpECTRE reuses SpEC's fit and update code verbatim (imported at runtime), so the *measurement* is at parity by construction.** Three things around it are not:

- the **two-stage Lev schedule** — stays in this issue (below);
- the **failure/abort conditions** — split out as a dedicated issue (the loop currently has no iteration cap and no divergence check);
- the **"smooth continuation"** at the end — the same defect as the PBJ branch losing its state, tracked with the PBJ issue #7413.

## The measurement — at parity, with one drift

- Both codes call the same `OmegaDotEccRemoval` functions. SpECTRE (`EccentricityControlParams.py:92-99`):

  ```python
  check_spec_import(contains_commit="ecfabf1c...")
  from OmegaDotEccRemoval import (ComputeOmegaAndDerivsFromFile, FindTmin, performAllFits)
  ```

  SpEC drives the same file from `Support/Perl/EccReduce.pm:245-247`.
- The fit windows match: `tmin` capped at 500 M, `tmax = tmin + 5π/Ω0` (2.5 orbits of data).
- **But the two codes pass different options to the shared code:**
  - SpEC production: `--no_check` (periastron-advance check off), varpro/freq-filter only on request, and a fit-model fallback from `F2cos2_SS` to `F2cos2` when `ecc > 0.01` — spin-spin terms overfit at large eccentricity (`EccReduce.pm:26-47`).
  - SpECTRE: hard-codes `opt_varpro=True`, `opt_freq_filter=True`, `check_periastron_advance=True`, takes whatever `performAllFits` returns (`EccentricityControlParams.py:188-215`).

  Deliberate improvements or accidental drift? A question, not a defect — open point 2.
- SpEC also has two non-zero-target variants (`InitialDataAdjustment.py`, `EOBEccControl.py`); SpECTRE asserts `target_eccentricity == 0.0`. Out of scope here; held by #5937.

## The iteration loop

**SpEC** (`Support/Perl/EccReduce.pm:114-433`):

- Each iteration runs **400 M + 2.5 orbits**, then creates the next `Ecc<N>` directory with updated `Omega0`/`adot0` and submits a fresh ID job.
- Convergence: `|ecc − target| < tol`.
- Two failure conditions: `MaxIts = 7` (`EccentricityReduction.cpp:18`, enforced `EccReduce.pm:373-388`) and "converging too slowly", with a same-Lev guard so a Lev switch is not mistaken for divergence (`EccReduce.pm:335-371`).

**SpECTRE** (`EccentricityControl.py:27-176`, re-entered via `Inspiral.yaml:19-37`):

- Same shape: `500 + 5π/Ω0` per iteration (SpEC uses 400 — unexplained drift, adjudicated in #7418); absolute tolerance, default `1e-3` (`InitialData.py:41-43`).
- **No iteration cap, no divergence check** (the dedicated issue).
- **One fixed Lev for the whole loop** (`TargetParams["EvolutionLev"]`, passed once at `PostprocessId.py:186`).

**The gap kept in this issue — no two-stage Lev schedule.** SpEC iterates cheaply at `MinLev` to `1e-3` ("RoughEccReduction"), then at `MaxLev` to `7e-4` (`DoMultipleRuns.input:30-46`). The rough→final switch also happens on both failure paths, so a stalled rough stage *promotes* rather than aborts (`EccReduce.pm:327-334,357-363,377-383`).

## Prior art

| # | state | relevance |
|---|---|---|
| **6460** | issue OPEN | duplicate of the Lev-schedule gap (empty body) — proposed closed as superseded by this issue |
| **6944** | PR merged | "EccControl: fix continuation after ecc-control is complete" — the current continuation behaviour; read before redesigning |
| **5966** | PR closed | abandoned 2024 attempt to port `OmegaDotEccRemoval` into SpECTRE — prior art for the SimulationSupport port; read why it was dropped |
| 6890 | PR draft | imports `OmegaDotEccRemoval` from SimulationSupport — **a module that does not exist there**; analysis in the #7412 survey comment |
| 6406 / 6295 / 6333 / 6467 / 6490 / 6468 + fixes | PR merged | built the automation loop, the SpEC dependency, the fit window, and the CLI this issue extends |
| 7089 | issue OPEN | nondeterministic pipeline-test failures — touches ecc-control tests |

Nothing existing covers the abort conditions — hence the dedicated issue.

## Proposed design

Port SpEC's two-stage schedule into the pipeline (pure pipeline change):

- `TargetParams` gains `RoughEccLev` (int, default `None` = rough stage disabled) and `RoughEccTolerance` (default `1e-3`); the existing `EccentricityAbsoluteTolerance` becomes the final-stage tolerance, default tightened to SpEC's `7e-4` when the rough stage is enabled.
- `EccentricityControl.py`: while the measured eccentricity is above `RoughEccTolerance`, iterate at `RoughEccLev`; once below, regenerate ID and continue iterating at the production Lev (`TargetParams["EvolutionLev"]`) to the final tolerance. The current stage is recorded in the eccentricity-params history file (`ecc_params_output_file`) so the loop is re-entrant.
- Interaction with the abort-conditions issue (split from this one): the divergence check compares only iterations at the same Lev (SpEC's same-Lev guard), and a rough-stage stall **promotes** to the final Lev instead of aborting (SpEC's failure-path behaviour).

**Testing / acceptance**: unit tests of the stage-switch logic on synthetic eccentricity histories — converges rough→final; stall promotes; behaviour unchanged when `RoughEccLev` is unset; the ecc-control pipeline test run once with the rough stage enabled.

## Open points to settle

1. [ ] **Knobs** — names (`RoughEccLev`/`RoughEccTolerance`) and the final-tolerance default (`7e-4` as in SpEC). Recommendation: adopt.
2. [ ] **Option drift on the shared fit** — keep SpECTRE's `varpro`/`freq_filter`/`check_periastron_advance` settings as deliberate improvements (recorded), or align with SpEC production (`--no_check`; `F2cos2` fallback at `ecc > 0.01`)? Must be recorded either way before the fit code moves to SimulationSupport, where both codes will share it.

A follow-up comment settling these points makes this issue ready for implementation (→ Ready).

---

🤖 drafted by: [Claude Fable 5](https://claude.com/claude-code)
(and reviewed by me 🙋)
