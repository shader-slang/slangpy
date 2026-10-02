# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# pyright: reportUnusedImport=false

from .fixtures import (
    benchmark_slang_function,
    BenchmarkSlangFunction,
    benchmark_python_function,
    BenchmarkPythonFunction,
    benchmark_compute_kernel,
    BenchmarkComputeKernel,
    report,
    ReportFixture,
)

# Re-exported so other projects can submit to BenchView with the same payload shape.
# A project supplies its own BenchViewProject; everything else is shared.
from .benchview import (
    build_benchview_observation,
    build_benchview_submissions,
    build_metric,
    submit_benchview_submissions,
    benchview_submission_url,
    BenchmarkSubmissionError,
    BenchViewMetric,
    BenchViewObservation,
    BenchViewProject,
    BenchViewSubmission,
    BENCHVIEW_MAX_BODY_BYTES,
)
from .utils import (
    get_gpu_infos,
    get_machine_info,
    get_commit_info,
    get_project_info,
)

# Clock control lives in the package, not in tools/, so a benchmark harness in
# another repository can import it rather than extend sys.path to reach it.
from .gpu_clock import (
    lock_gpu_clocks,
    unlock_gpu_clocks,
)
