# Image-processing shader coverage

This example uses the experimental coverage API to run an edge-preserving 3x3
denoiser, HDR normalization, and display gamma. It generates its own noisy image
with NumPy and writes PNG previews using SlangPy, so no extra packages, image
downloads, or Slang source checkout are needed.

From the SlangPy repository root, after building the PR:

```sh
python -m examples.shader_coverage.main --device vulkan --output-dir shader-coverage-report
```

Counters default to **64 bits**. On MoltenVK or another device without 64-bit
buffer atomics, explicitly add `--counter-width 32`. Use `--device cuda` to try
the CUDA backend on Windows/Linux; CUDA execution is pending validation for the
draft. D3D12 coverage is not implemented.

Open `shader-coverage-report/index.html` to see image pairs, before/after coverage
totals, and expandable source listings with branch hit counts. The two JSON
captures preserve raw metadata and counters, with counter values encoded as
decimal strings. `summary.json` contains coverage and numerical validation.
These files are an example-specific format, not a public snapshot serialization
API. The small HTML writer is local to this example; reusable reporting and
offline merging remain follow-up work.

The three scenarios reuse one compiled program:

1. An ordinary opaque image with denoising enabled.
2. An HDR image with black and transparent regions.
3. The HDR image with denoising disabled.

The ordinary image misses the zero-light guard, HDR normalization, transparent
early return, and denoising bypass. The targeted inputs cover those paths. The
script checks that coverage improves and every instrumented line and branch arm
in `postprocess.slang` is hit. It also compares every result against NumPy and
uninstrumented GPU execution, checks pixel invocation counts, and verifies that
the final capture resets the counters.

The API calls to look for in `main.py` are:

```python
compiler_options={"coverage": spy.ShaderCoverageOptions(counter_width=args.counter_width)}
snapshot = device.shader_coverage.snapshot()             # cumulative
snapshot = device.shader_coverage.snapshot(reset=True)   # capture and clear
```

The report compares cumulative snapshots instead of adding them together.
Branch IDs remain scoped to the single program generation; it never merges
branches using source coordinates alone. Only the example shader contributes to
the displayed totals. Generated wrappers and library entries remain in the raw
manifest. Coverage describes compiled instrumented sites, not uncompiled code.
