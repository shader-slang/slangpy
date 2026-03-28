# Phase 4: Finalize

1. **Run pre-commit** — `pre-commit run --all-files`
2. **Build and test** — `cmake --build --preset windows-msvc-debug && python tools/ci.py unit-test-cpp`
3. `pre-commit run --all-files` — passes (re-run if it modifies files)

Note: No Python bindings in this phase. The BC codec is C++ only.
