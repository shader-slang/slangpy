# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Command line entry point for GPU clock control.

The implementation lives in ``slangpy.testing.benchmark.gpu_clock`` so that
benchmark harnesses, including ones in other repositories, can import it instead
of reaching into this directory through ``sys.path``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slangpy.testing.benchmark.gpu_clock import main  # noqa: E402

if __name__ == "__main__":
    main()
