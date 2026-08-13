Merging into #7413: the feature-parity survey (2026-08-13) found the two
issues are one code path —
`support/Pipelines/Bbh/EccentricityControl.py:120-140` →
`start_inspiral(..., id_subfile_name="PostJunkVolumeData")` →
`support/Pipelines/Bbh/Inspiral.yaml:50-64,143-170`. Fixing the Lev
workflow (#7413) alone multiplies the state defects across N Levs; fixing
the state (this issue) alone leaves the branch producing a single Lev.

The full evidence for this issue's scope — hard-coded
`InitialTimeStep`/`InitialSlabSize`, discarded multistep history, and the
control-system state reset (`simple_tags_from_options`) — is in the survey
comment on #7413 (gap G3), together with the design questions (checkpoint
restart vs volume-data import; which state crosses the branch; #6849).

---

🤖 drafted by: [Claude Fable 5](https://claude.com/claude-code)
(and reviewed by me 🙋)
