# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import pathlib


def slang_include_paths() -> list[pathlib.Path]:
    return [pathlib.Path(__file__).parent / "slang"]
