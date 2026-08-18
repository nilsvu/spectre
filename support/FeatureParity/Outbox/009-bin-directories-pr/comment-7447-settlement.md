Settlement addendum after co-review of the implementation (see the linked PR): three points revised toward simpler code.

- **3 (Python/CLI snapshot) — narrowed.** The bin directory freezes spectre's own package and CLI; third-party Python packages are **not** frozen — they come from the machine environment at job time. Evidence for the relaxation: the `BOOTSTRAP_PY_DEPS` path installed packages where nothing looked for them on Debian-patched pips, and nobody noticed — no scheduled workflow depends on frozen bootstrapped dependencies. That bug is filed as its own issue with the measurements; this PR leaves dependency bootstrapping untouched. Freezing bootstrapped dependencies is deferred, not tracked yet.
- **5 (machine environment) — deferred to #7443.** No environment file is copied into the bin directory; jobs keep sourcing what they source today. #7443's machine-directory design owns the machine→env-script mapping; a mapping here would be double work.
- **Contents, further narrowed:** the minimal `Manifest.yaml` from the design is dropped as not load-bearing — provenance stays with formaline, which already embeds the source archive and environment in every executable and H5 output.
- **6 (self-containment enforcement) — implemented as a build-shape check.** The hard error stands, detected as shared SpECTRE libraries in the build's `lib/` (what `BUILD_SHARED_LIBS=ON` produces) instead of per-executable `ldd` parsing.

Deferred and not yet tracked: the deliberate-update path (point 4, as settled) and bootstrapped-dependency freezing (above).

---

🤖 drafted by: [Claude Fable 5](https://claude.com/claude-code)
(and reviewed by me 🙋)
