# Compiler capability probe

This standalone research tool calls the Slang session API directly. It does not modify SlangPy's production API or defaults. It exists to inspect capability/profile/downstream interactions before implementing the migration.

See [measured results](../../plan/compiler-capabilities-probe-results.md), [implementation plan](../../plan/compiler-capabilities.md), and [workaround ledger](../../plan/slang-compiler-workarounds.md).

Build SlangPy before collecting device inventories or running its smoke shaders. Then configure/build this tool with the headers for the compiler API being investigated:

```powershell
cmake --build --preset windows-msvc-debug
cmake -S tools/compiler_capability_probe -B build/compiler-capability-probe -G Ninja -DSLANG_INCLUDE_DIR=C:/projects/slangpy/build/windows-msvc/_deps/slang-src/include
cmake --build build/compiler-capability-probe
```

The native executable loads the library passed with `--library`; it does not link a compiler DLL implicitly. Use compatible public API headers. The Python runner creates one process per experiment and records the actual loaded compiler path/build tag. `--dxc` accepts its library or containing directory; on Windows `--nvrtc` accepts the NVRTC DLL filename. The tool translates these to the different path conventions required by Slang's downstream locators.

```powershell
python tools/compiler_capability_probe/run.py --probe build/compiler-capability-probe/compiler_capability_probe.exe --library build/windows-msvc/Debug/slang-compiler.dll --output build/compiler-capability-probe/results/release --dxc build/windows-msvc/Debug/dxcompiler.dll --nvrtc "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/bin/nvrtc64_120_0.dll"
```

This collects device capabilities and executes a buffer-write/readback smoke shader on available SlangPy backends. Compare another compiler with `--inventory <first-run>/device_inventory.json` to hold the input device list fixed. `--filter <substring>` selects cases for diagnosis; use a separate output directory for filtered runs because `results.json` describes only that invocation.

The runner's exit status reports harness errors, not compiler rejection: many cases intentionally fail. Verify the recorded baseline separately:

```powershell
python tools/compiler_capability_probe/verify.py build/compiler-capability-probe/results/release build/compiler-capability-probe/results/master
```

The verifier describes the observed September 2026 Windows/NVRTC 12.2/CC 7.5 baseline. Some expectations are deliberately hardware/toolkit-specific. Future compiler fixes should change its results and trigger updates to the workaround ledger; this is not a generic cross-platform CI gate.

Each case directory retains `input.slang`, `command.json`, `stdout.txt`, `stderr.txt`, `result.json`, and successful generated output in `output.bin`. Despite its suffix, that output is readable text for HLSL, DXIL assembly, PTX, Metal, WGSL, and C++; SPIR-V is binary. `environment.json` records library hashes and supported NVRTC architectures.

Generated code for an architecture newer than the local GPU is not executed. Metal/WGSL source generation does not imply downstream or runtime validation. The current native loader and output inspection have been exercised on Windows; the POSIX loader is provided but untested in this investigation.
