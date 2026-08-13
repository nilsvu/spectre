Re-scoping note from the feature-parity survey (2026-08-13): the plan of record for initial orbital parameters is now the `sxs-collaboration/SimulationSupport` package (#7412), which carries a byte-compatible copy of the current SpEC PN kernels — its test asserts the same numbers SpECTRE asserts today, so adoption is a pure refactor.

This PR is not compatible with that plan as written: it **changes the PN numbers** (at `separation=16.0`, `Omega_0` 0.014474280975952748 → 0.014454484323416913; at `NumOrbits=20`, `D_0` 16.042 → 15.711). The GPR-fitted guesses being added to SimulationSupport compound this — they are trained on residuals *relative to the SpEC PN baseline* (`SimulationSupport/gpr/`, baseline `ZeroEccParamsFromPN.omegaAndAdot` at `rPrime0 = 1`), so changing the baseline invalidates the fits.

Proposal: re-scope this PR's idea to "improve the PN kernels inside SimulationSupport" — i.e. PostNewtonian.jl (or an updated closed-form implementation) as a SimulationSupport-side change, coordinated with retraining the GPR residuals, rather than a SpECTRE-side replacement. The SpECTRE-side wiring slot goes to the SimulationSupport import (#7412, PR #6890's initial-orbital-parameters half).

---

🤖 drafted by: [Claude Fable 5](https://claude.com/claude-code)
(and reviewed by me 🙋)
