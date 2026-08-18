Implementation note from #7447 (bin directories), at sxs-collaboration/spectre@06fa7dffb1: `support/Pipelines/Bbh/EccentricityControl.py:138` passes `pipeline_dir=lev_dir.path`, so each Lev branch becomes its own pipeline directory — and with #7447's snapshot, each Lev gets its own bin directory with its own copies of the executables and the Python package.

That is correct under #7447's settled design (one bin directory per pipeline directory), but it contradicts the intent recorded in the #5951 discussion ("we want 1 bin directory for all Ecc iterations, levs, segments"), and the disk cost scales with the number of Levs.

When settling this issue, consider having the Lev branches share the parent pipeline directory's bin — they are branches of one simulation, not separate simulations.

---

🤖 drafted by: [Claude Fable 5](https://claude.com/claude-code)
(and reviewed by me 🙋)
