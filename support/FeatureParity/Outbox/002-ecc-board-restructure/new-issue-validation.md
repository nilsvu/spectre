Split out of #7418 (feature-parity survey, 2026-08-13; revisions SpEC @
`5f8f5375ca`, SpECTRE `develop` @ `4d43624d64`). #7418 keeps the bounded
documentation task (document intentional gauge/constraint-damping
differences); this issue is the validation programme for the schemes where
"match SpEC's parameters" is not a meaningful frame.

**Where SpEC and SpECTRE differ structurally, parameter comparison cannot
establish parity — accuracy at fixed cost can.** Three areas (full
`file:line` enumeration in the #7418 survey comment):

1. **Time stepping.** SpEC: `DormandPrince5`, global adaptive
   (`AdaptiveDense` + `ProportionalIntegral`,
   `InputFiles/Bbh/Evolution.input:213-221`, `ODETolerance = 1e-8`).
   SpECTRE: `AdamsMoultonPcMonotonic` order 4 with local time stepping and
   `ErrorControl` (`support/Pipelines/Bbh/Inspiral.yaml:226-244`,
   tolerances `1e-10`/`1e-8` with the comment that 100x smaller tolerances
   "reduced the noise in the constraints significantly").
2. **Filtering / spatial discretization.** SpEC: multi-domain spectral with
   exponential filters (definitions live in domain/subdomain input, not yet
   fully located — the #7418 survey searched `InputFiles/Bbh/*.input`).
   SpECTRE: DG with `Hypercube HalfPower: 420` filtering
   (`Inspiral.yaml:287-303`). The comparable quantity is effective
   dissipation, not option values.
3. **Resolution / AMR.** SpEC: real AMR driven by
   `TruncationErrorMax = 0.000216536·4^(-k)`
   (`DoMultipleRuns.input:823-843`). SpECTRE: Levs are pure p-refinement
   (`Inspiral.py:26-33`) and the `Amr:` block has an empty `Criteria:`
   (`Inspiral.yaml:313-324`) — configured but inert. (The AMR gap itself
   belongs to the AMR work, not here; what belongs here is the accuracy
   comparison once both codes can run the same configuration.)

Deliverable shape: pick the comparison configuration (issue #5133, "Choose
BBH configuration for comparison with SpEC", is the setup issue this
depends on), run matched SpEC/SpECTRE simulations, and compare waveform
accuracy and constraint levels at fixed cost. Success criteria should be
preregistered per run.

---

🤖 Drafted with [Claude Code](https://claude.com/claude-code) as the feature-parity campaign survey agent; reviewed and posted by a human.
