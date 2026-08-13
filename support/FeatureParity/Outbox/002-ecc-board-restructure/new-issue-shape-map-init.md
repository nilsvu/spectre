Split out of #7417 (feature-parity survey, 2026-08-13; revisions SpEC @
`5f8f5375ca`, SpECTRE `develop` @ `4d43624d64`). #7417 keeps the
re-initialization / failure-recovery design; this issue is the small,
fully-specified gap inside it.

**Initialize the shape map from the measured ID horizon coefficients
instead of an analytic Kerr horizon.**

- SpEC initializes shape/size maps from the **measured** horizon of the
  initial-data solve — `InputFiles/Bbh/SpatialCoordMap.input:26-30`
  (`ShapeACpFile = ID_Init_FuncLambdaFactorA.txt;` etc., produced alongside
  `ID_AhACoefs.dat`/`ID_AhBCoefs.dat`).
- SpECTRE uses an **analytic** Kerr horizon from mass and spin —
  `support/Pipelines/Bbh/Inspiral.yaml:180-190`
  (`KerrSchildFromBoyerLindquist`,
  `src/Domain/Creators/TimeDependentOptions/ShapeMap.hpp:31-53`), with the
  values taken from the ID *input* file (`Inspiral.py:268-275`), not from
  the ID solve's horizon output.
- SpECTRE **already has** the SpEC-equivalent option and it is unused:
  `YlmsFromFile` (`ShapeMap.hpp:64-114`), listed in the `InitialValues`
  variant at `ShapeMap.hpp:272-276` (also `YlmsFromSpEC`,
  `ShapeMap.hpp:134-165`).
- The pipeline **already computes the arguments it would need and throws
  them away**: `support/Pipelines/Bbh/Inspiral.py:265-267` sets
  `HorizonsFile`, `AhASubfileName`, `AhBSubfileName` — grep over
  `support/Pipelines/` shows no template consumes them.

So this is dead plumbing that can be finished independently: wire the three
computed arguments to a `YlmsFromFile` block in `Inspiral.yaml`.

Two things to check while doing it:

1. SpEC records a consistency constraint between the excision radius and
   the ID horizon (`DoMultipleRuns.input:382-387`: the `rExc`/`rInitAh`
   ratio must match what initial data assumed, "or else shape control will
   not have Q=0 at the start of the evolution") — verify SpECTRE's
   equivalent relation holds when the shape map comes from the measured
   horizon.
2. This may remove most of the motivation for the larger #7417 design:
   SpEC's re-initialization recovery path exists precisely because a wrong
   shape-map initialization can kill the run in the first ~10 M
   (`Support/Perl/BatchJobTermination.pm:295-302,757-823`). After this
   lands, re-measure whether the recovery path is still needed.

Prior art: PR #6113 shipped the `InitialValues` variant options; no issue
or PR covers actually using `YlmsFromFile` in the BBH pipeline (dedupe
recorded in the #7417 survey comment).

---

🤖 drafted by: [Claude Fable 5](https://claude.com/claude-code)
