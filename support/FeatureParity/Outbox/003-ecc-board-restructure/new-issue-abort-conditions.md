Split out of #7416 (feature-parity survey, 2026-08-13; revisions SpEC @ `5f8f5375ca`, SpECTRE `develop` @ `4d43624d64`).

**The eccentricity-control loop has no abort conditions.** This is a standalone correctness/cost problem with a bounded fix, independent of both the Lev schedule (#7416) and the continuation mechanism (#7413).

`support/Pipelines/Bbh/EccentricityControl.py:27-176` re-enters itself through the `Next:` block of the inspiral input file (`support/Pipelines/Bbh/Inspiral.yaml:19-37`) with **no iteration cap and no divergence check**. An unconverging ecc-control loop currently regenerates initial data and resubmits indefinitely — it runs until the allocation is gone.

SpEC has both conditions:

- **Maximum iterations**: `MaxIts` default 7 (`Utils/DataBox/EccentricityReduction.cpp:18`), enforced at `Support/Perl/EccReduce.pm:373-388`. During rough (low-Lev) reduction, hitting the cap promotes to the final Lev instead of aborting (`:377-383`).
- **"Eccentricity converging too slowly"**: `Support/Perl/EccReduce.pm:335-371` — abort when `abs(Ecc-TargetEcc) > abs(OldEcc-OldEccT)`, i.e. the eccentricity did not improve over the previous iteration. The same-Lev guard at `:350-351` exists precisely so that a Lev switch is not mistaken for divergence — a detail to keep if #7416's two-stage schedule is implemented.

The numbers to port: `MaxIts = 7`; divergence = "no improvement over the previous iteration, comparing iterations at the same Lev".

**No prior art exists** — no SpECTRE issue or PR covers either condition (dedupe pass over ecc-control issues/PRs recorded in the #7416 survey comment).

## Proposed implementation

Pure pipeline change in `EccentricityControl.py`:

- `TargetParams` gains `MaxEccIterations` (default 7, SpEC's `MaxIts`).
- The iteration count and per-iteration `(Lev, eccentricity)` come from the eccentricity-params history the pipeline already accumulates (`ecc_params_output_file`); no new state file.
- In the non-converged branch, before regenerating ID:
  1. if `iterations >= MaxEccIterations`: stop with an error that prints the full iteration history;
  2. if `|ecc − target|` did not decrease relative to the previous iteration *at the same Lev* (SpEC's same-Lev guard): stop with an error naming both values.
- On stop, everything stays on disk; the error message states how to resume manually with adjusted parameters. Once #7416's rough stage exists, a rough-stage stall promotes to the final Lev instead of stopping (SpEC's behaviour).

**Testing / acceptance**: unit tests on synthetic histories — cap reached, divergence at the same Lev, Lev switch not flagged as divergence, clean convergence unaffected.

## Open points to settle

1. [ ] **Iteration cap** — adopt SpEC's default of 7 for `MaxEccIterations`?
2. [ ] **Divergence response** (before #7416's rough stage exists) — hard stop with error (recommendation) vs warn-and-continue.

A follow-up comment settling these points makes this issue ready for implementation (→ Ready).

---

🤖 drafted by: [Claude Fable 5](https://claude.com/claude-code)
(and reviewed by me 🙋)
