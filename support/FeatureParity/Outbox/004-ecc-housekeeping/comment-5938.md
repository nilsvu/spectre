Update from the feature-parity survey (2026-08-14): @nilsdeppe's suggestion above is adopted. #7415's design now covers this issue's ask as its "Part A": compute `T(N orbits)` from PN (the same T4 integrator behind `initial_orbital_parameters`, moving to SimulationSupport in #7412) at input-file generation and terminate with the existing `TimeCompares` machinery — a pure pipeline change, no orbit-count trigger. The ecc-control pipeline already runs the constant-Ω version of this (`FinalTime = 500 + 5π/Ω0`).

The orbit-*phase* trigger survives only for wave-extraction cadence (N times per orbit, SpEC's production 400/orbit), where a PN precompute would drift from the actual evolution — that is #7415's "Part B", with PR #6009 as the starting point.

So: this issue is resolved by #7415's Part A once implemented; tracking continues there.

---

🤖 drafted by: [Claude Fable 5](https://claude.com/claude-code)
(and reviewed by me 🙋)
