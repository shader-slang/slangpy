# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generate binding docstrings using the repository's vendored pybind11_mkdoc."""

from pathlib import Path
import sys


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "external"))
    from pybind11_mkdoc import main

    sys.exit(main())
