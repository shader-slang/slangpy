# Image-processing shader coverage

This example measures a multi-file GPU pipeline: an edge-preserving 3x3 denoiser,
automatic-differentiation-based exposure adjustment, and runtime interface
dispatch between two tone mappers. It generates its own noisy image
with NumPy and writes PNG previews using SlangPy, so no extra packages or image
downloads are needed.

From the SlangPy repository root, after building the PR:

```sh
python -m examples.shader_coverage.shader_coverage --slang-source /path/to/slang --device vulkan --output-dir shader-coverage-report --open
```

Add `--no-coverage` to disable instrumentation and collection while running the
same image-processing workload. The report then contains images and numerical
validation only, marks coverage as disabled, and has no coverage-report links.
This overrides `--coverage`, `--boolean`, and `--count`; `--slang-source` is unused in this mode.
In Python, the switch is simply `compiler_options={"coverage": None}` versus
`compiler_options={"coverage": spy.ShaderCoverageOptions(...)}`, set before shaders
are loaded. Snapshot calls run only when coverage is enabled.

All three coverage kinds are enabled by default. Select one or more with
`--coverage line`, `--coverage branch function`, or `--coverage line branch function`.
The landing page marks omitted kinds as disabled. For example:

```sh
python -m examples.shader_coverage.shader_coverage --coverage branch --boolean --counter-width 32 --slang-source /path/to/slang
```

Add `--boolean` to measure hit/miss using non-atomic stores
instead of counting executions. This avoids atomic contention; it does not preserve execution
frequencies. Execution counting is the default (`--count`). The landing
page labels the recording mode. Repeated executions
leave each hit at 1 until reset. Use `--boolean --counter-width 32` for 32-bit
slots; boolean mode does not pack counters into bits.

This workload requires Slang **2026.19 or newer**, included by this PR. That
release fixes atomic coverage probes inside differentiated functions, allowing
both counting and boolean recording with every coverage kind. Coverage in
generated derivatives follows executed source-mapped probes; it does not promise
primal-call frequencies or separate metrics for each generated derivative.

Counters default to **64 bits**. On MoltenVK or another device without 64-bit
buffer atomics, explicitly add `--counter-width 32` for count mode. Boolean mode
requires integer support for its slot width, not atomic support. Use `--device cuda` for
the CUDA backend on Windows/Linux. Vulkan and CUDA have been validated on a
Windows RTX 4090, including 64-bit counters. D3D12 coverage is not implemented.

The runner uses Python on Windows, Linux, and macOS. The optional `--slang-source` points
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
HTML includes all five example shader modules; the LCOV files retain all source
entries
from the captured program. The two JSON
captures preserve the source text of every workload module, raw metadata and
counters, with counter values encoded as
decimal strings. `summary.json` contains coverage and numerical validation.
These files are an example-specific format, not a public snapshot serialization
API. Conversion and coverage rendering use Slang's existing tools; the landing
page with images is local to this example. A reusable SlangPy reporting API and
offline merging remain follow-up work.

Without `--slang-source`, the same entry point creates images, raw captures,
and a landing page with coverage totals. Annotated coverage reports and LCOV
exports require the Slang tools:

```sh
python -m examples.shader_coverage.shader_coverage --device vulkan --output-dir shader-coverage-report
```

The three scenarios reuse one compiled program:

1. An ordinary opaque image with denoising enabled.
2. An HDR image with black and transparent regions.
3. The HDR image with denoising disabled and the Reinhard tone mapper selected.

The ordinary image misses the zero-light guard, HDR normalization, transparent
early return, denoising bypass, and the Reinhard implementation. The targeted inputs cover those paths.
Numerical validation compares each result against NumPy and ordinary GPU
execution. `test_shader_coverage_example.py` checks coverage progression, per-file coverage, runtime dispatch to both implementations, numerical results,
and reset behavior.

The shader is split into five files:

- `postprocess.slang` composes the per-pixel pipeline and handles transparency.
- `coverage_color.slang` supplies shared luminance calculation.
- `coverage_denoise.slang` implements the edge-preserving neighborhood filter.
- `coverage_exposure.slang` marks `exposureLoss` as differentiable, calls
  `fwd_diff` to obtain its exposure derivative, and uses that derivative in a
  bounded gradient step toward middle gray. NumPy independently computes the
  analytic derivative for validation.
- `coverage_tonemap.slang` defines `IToneMapper`, `NormalizeToneMapper`, and
  `ReinhardToneMapper`. `createDynamicObject` selects the implementation using a
  runtime ID. Python registers both `TypeConformance` entries once, so changing
  the ID does not create a separate program or coverage generation. Both methods
  are named `map`; the per-file report retains their distinct source locations.

The first capture misses the Reinhard implementation; the expanded capture
executes both. The generated derivative executes source-mapped probes belonging
to `exposureLoss`; this reports source coverage, not separate derivative-IR sites.
The image workload is unchanged whether instrumentation is on or off.

The Python code is separated by responsibility:

- `shader_coverage.py` is the runnable example. Read it to see device configuration,
  ordinary workload calls, and the two capture boundaries.
- `image_processing.py` generates inputs, loads and runs the shader, saves image
  previews, and validates numerical results. It has no coverage logic.
- `report.py` exports snapshots and builds the image landing page and optional
  LCOV reports. It can be imported without loading SlangPy or a GPU runtime.

The essential pattern in `shader_coverage.py` is:

```python
with spy.Device(
    type=spy.DeviceType.vulkan,
    compiler_options={"coverage": spy.ShaderCoverageOptions()},
) as device:
    processor = ImageProcessor(device)
    processor.process(ordinary_image)
    before = device.shader_coverage.snapshot()
    processor.process(hdr_image)
    processor.process(hdr_image, denoise=False, mapper_id=1)
    after = device.shader_coverage.snapshot(reset=True)
# Both snapshots remain usable after device close.
```

Instrumentation must be enabled before loading shaders. `ImageProcessor` accepts
an ordinary device too: its shader and dispatch code do not change. The example
uses explicit snapshot calls so the measurement boundaries remain visible;
a reusable coverage wrapper is left for a follow-up PR.

Direct execution also follows the other examples' naming convention:

```sh
python examples/shader_coverage/shader_coverage.py --device vulkan
```

The report compares cumulative snapshots instead of adding them together.
Branch IDs remain scoped to the single program generation; it never merges
branches using source coordinates alone. Only the five example shader modules contribute to
the displayed totals. The landing page includes a per-file breakdown, and source
text is checked for every module before rendering. Generated wrappers and library entries remain in the raw
manifest. Coverage describes compiled instrumented sites, not uncompiled code.
