# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Importing these here means we don't have to explicitly import them into every benchmark
# file, even if the benchmark file doesn't directly reference them.
from slangpy.testing.benchmark import benchmark_slang_function, benchmark_python_function, benchmark_compute_kernel, report  # type: ignore

pytest_plugins = ["slangpy.testing.plugin", "slangpy.testing.benchmark.plugin"]
