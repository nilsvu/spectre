Closing as completed: the CLI exists — `spectre bbh eccentricity-control`
(`support/Pipelines/Bbh/__init__.py:14,24-27`,
`support/Pipelines/Bbh/EccentricityControl.py`), shipped by PR #6468 and
the surrounding ecc-control automation (PRs #6406, #6295). It manages the
outer loop of eccentricity reduction as this issue asked, calling SpEC's
fit code at runtime.

The follow-up work this issue anticipated is tracked elsewhere: replacing
the SpEC runtime dependency via SimulationSupport (#7412 for the initial
orbital parameters, #7416 for the measurement side), and the remaining loop
gaps (#7416: two-stage Lev schedule; abort conditions split into a
dedicated issue).

---

drafted by: Claude Fable 5
