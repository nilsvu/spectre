Scope note from the feature-parity survey (2026-08-13): the zero-target
parts of this tracker are now covered by granular board issues — #7412
(initial orbital parameters, with the SimulationSupport plan and the
GPR-fitted guesses), #7416 (reduction iterations and Lev schedule), #7413
(PBJ continuation), plus a dedicated abort-conditions issue.

Proposal: narrow this issue to what no successor covers —

- **nonzero target eccentricity** (SpEC: `InitialDataAdjustment.py` for
  orbital-quantity control to a target, `EOBEccControl.py` for EOB
  waveform-based control; SpECTRE asserts `eccentricity == 0.0` in
  `EccentricityControlParams.py:113-115` and
  `InitialOrbitalParameters.py:85-88`, while `TargetParams` already carries
  a `MeanAnomalyFraction` that nothing reads —
  `support/Pipelines/Bbh/InitialData.py:34`);
- **BNS/BHNS orbital parameters** (SpEC's ID scripts shell out to
  `ZeroEccParamsFromPN` — `Support/Python/GrHydro_ID_script_functions.py:57,92`).

Whether eccentric-orbit support is in the feature-parity campaign's scope
is an open question; until decided, this issue is the holder for it.

---

drafted by: Claude Fable 5
