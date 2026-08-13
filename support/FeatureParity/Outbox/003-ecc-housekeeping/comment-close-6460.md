Closing as superseded by #7416, which tracks the same gap with survey
evidence: SpEC's `RoughEccReduction` two-stage Lev schedule
(`InputFiles/Bbh/DoMultipleRuns.input:30-46` — iterate at `MinLev` to
`1e-3`, then at `MaxLev` to `7e-4`; switch at
`Support/Perl/EccReduce.pm:327-334`). See the survey comment on #7416 for
the full mechanism comparison.
