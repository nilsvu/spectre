Implementation note from #7447 (bin directories), at sxs-collaboration/spectre@06fa7dffb1: `support/Pipelines/Bbh/EccentricityControl.py:138` passes `pipeline_dir=lev_dir.path`, making each Lev branch its own pipeline directory.

#7447's implementation resolves the bin directory by nearest-ancestor search, bounded to the simulation directory formats in `support/Python/DirectoryStructure.py` — a nested Lev branch finds and reuses the parent pipeline directory's bin instead of creating its own. All Levs and ecc iterations therefore share one bin directory (the intent recorded in #5951), and settling this issue needs no bin-directory work.

One interaction to keep in mind: `--no-create-bin` at pipeline start propagates through `Next`, so a simulation that opted out stays opted out across Lev branches.

---

🤖 drafted by: [Claude Fable 5](https://claude.com/claude-code)
(and reviewed by me 🙋)
