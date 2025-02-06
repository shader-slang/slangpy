# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import pathlib
import subprocess

root_dir = pathlib.Path(__file__).parent.parent

docs_dir = root_dir / "docs"
docs_build_dir = docs_dir / "_build"

subprocess.call(f"python -m sphinx -b html {docs_dir} {docs_build_dir}")
