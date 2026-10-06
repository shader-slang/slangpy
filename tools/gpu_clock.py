#!/usr/bin/env python

# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Command line entry point for GPU clock control.

The implementation lives in ``slangpy.testing.benchmark.gpu_clock`` so that
benchmark harnesses, including ones in other repositories, can import it instead
of reaching into this directory through ``sys.path``.
"""

import runpy
from pathlib import Path

if __name__ == "__main__":
    # Cleanup must work even when the native extension or pytest cannot import.
    # Execute the shared stdlib-only module without importing slangpy.__init__.
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "slangpy/testing/benchmark/gpu_clock.py"),
        run_name="__main__",
    )
