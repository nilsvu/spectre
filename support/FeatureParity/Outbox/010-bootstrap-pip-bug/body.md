# BOOTSTRAP_PY_DEPS installs packages where nothing looks for them on Debian-patched pips

At sxs-collaboration/spectre@06fa7dffb1: `cmake/BootstrapPyDeps.cmake:30-39` runs `pip install --prefix <build_dir>`, and `CMakeLists.txt:86-89` puts `<build_dir>/lib/pythonX.Y/site-packages` on the `PYTHONPATH` for the wrappers, `LoadPython.sh`, and the test environments.

**On Debian-patched pips those two paths disagree, so bootstrapped dependencies are downloaded and then never found.** Measured with pip 22.0.2 (Ubuntu-derived): `pip install --prefix P` lands in `P/local/lib/python3.10/dist-packages`, not `P/lib/python3.10/site-packages`. Nothing errors — configure succeeds, the download happens, and every later import falls back to whatever the machine environment provides (or fails).

Also measured, for whoever fixes this: `--prefix` respects already-satisfied requirements ("Requirement already satisfied" for packages importable via the environment; only missing ones are installed), so the fix is only about *where the result lands*, not about re-fetching.

**Fix direction:** make the `PYTHONPATH` point at the directory pip actually used instead of the assumed one — either query the interpreter's install scheme (`sysconfig`) for the `--prefix` layout, or discover the non-empty directory under `<build_dir>/{local/,}lib/python*/{site,dist}-packages` after the install and error on ambiguity. A few lines in `cmake/BootstrapPyDeps.cmake` / `CMakeLists.txt`.

Found while implementing #7447 (which deliberately leaves dependency bootstrapping untouched).

---

🤖 drafted by: [Claude Fable 5](https://claude.com/claude-code)
(and reviewed by me 🙋)
