Consolidated into #7447, which now carries the full SpEC-vs-SpECTRE survey and a design proposal for run-local bin directories. The constraints from this thread are folded into that design and its open points: one bin directory per simulation rather than per segment, executables copied with a self-containment check instead of copying shared libraries, and an explicit path to deliberately update code mid-simulation. Please follow up on #7447.

---

🤖 drafted by: [Claude Fable 5](https://claude.com/claude-code)
(and reviewed by me 🙋)
