# Image-processing shader coverage

This example uses the experimental coverage API to run an edge-preserving 3x3
denoiser, HDR normalization, and display gamma. It generates its own noisy image
with NumPy and writes PNG previews using SlangPy, so no extra packages, image
downloads are needed.

From the SlangPy repository root, after building the PR:

```sh
python -m examples.shader_coverage.run --slang-source /path/to/slang --device vulkan --output-dir shader-coverage-report --open
```

Counters default to **64 bits**. On MoltenVK or another device without 64-bit
buffer atomics, explicitly add `--counter-width 32`. Use `--device cuda` for
the CUDA backend on Windows/Linux. Vulkan and CUDA have been validated on a
Windows RTX 4090, including 64-bit counters. D3D12 coverage is not implemented.

The runner uses Python on Windows, Linux, and macOS. `--slang-source` must point
to a Slang checkout containing `tools/shader-coverage/slang-coverage-to-lcov.py`
and `tools/coverage-html/slang-coverage-html.py`; no Slang compiler build is
required for these tools. The installed SlangPy still supplies the compiler used
to execute the example. On Windows with CUDA, set `CUDA_PATH` to your toolkit
directory if the compiler cannot find NVRTC, for example:

```powershell
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0'
```

Open `shader-coverage-report/index.html` to see image pairs, before/after coverage
totals, and links to Slang-rendered LCOV reports for the ordinary input and the
expanded inputs. Each report links to annotated shader source with line,
function, and branch counts. The landing page also links to the `.info` LCOV
files for other LCOV-compatible tools. `--open` opens this page in your browser;
without it, the runner prints the path and browser URL. The output directory
is a static report that can be copied or shared as a whole.

The LCOV files live under `coverage/ordinary_only/` and
`coverage/expanded_inputs/`, with rendered pages under each `html/` directory.
HTML is filtered to `postprocess.slang`; the LCOV files retain all source entries
from the captured program. The two JSON
captures preserve raw metadata and counters, with counter values encoded as
decimal strings. `summary.json` contains coverage and numerical validation.
These files are an example-specific format, not a public snapshot serialization
API. Conversion and coverage rendering use Slang's existing tools; the landing
page with images is local to this example. A reusable SlangPy reporting API and
offline merging remain follow-up work.

To run without a Slang source checkout, the original command still creates
images, raw captures, and a basic HTML page with expandable source listings:

```sh
python -m examples.shader_coverage.main --device vulkan --output-dir shader-coverage-report
```

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
