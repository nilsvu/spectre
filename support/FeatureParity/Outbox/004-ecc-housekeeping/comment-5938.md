Linking from the feature-parity survey (2026-08-13): this issue is the use
case; the trigger mechanism is tracked in #7415, which now carries the full
SpEC-vs-SpECTRE evidence (SpEC's `FractionOfOrbit` is capped at
`frac <= 0.25`, so "every N orbits" has no SpEC counterpart and is a
SpECTRE-only design).

PR #6009, written against this issue, is a substantially complete
implementation with one known bug (the dense-trigger `next_check_time`
logic — the author's own diagnosis). Proposal: make #7415 the
implementation issue and revive or explicitly supersede #6009 there rather
than reimplementing it silently.

---

🤖 drafted by: [Claude Fable 5](https://claude.com/claude-code)
(and reviewed by me 🙋)
