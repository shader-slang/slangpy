# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python macro tool.
"""

import sys
import re
from pathlib import Path
from enum import Enum
from io import StringIO
from typing import Any


class Capturing(list):  # type: ignore (no idea why this inherits from none-typed list!)
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args: Any):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


HEADER_START_RE = re.compile(r"^\s*\/\*\s*<<<PYMACRO\s*$")
HEADER_END_RE = re.compile(r"^\s*>>>\s*\*\/\s*$")
FOOTER_RE = re.compile(r"^\s*\/\*\s+<<<PYMACROEND>>>\s*\*\/\s*$")


class State(Enum):
    IDLE = 1
    HEADER = 2
    CONTENT = 3


def process_file(path: Path):

    state = State.IDLE
    script_lines = []

    # print(path)
    lines_in = open(path).readlines()
    lines_out = []

    for line_index, line in enumerate(lines_in):
        # print(line_index, line)
        if state == State.IDLE:
            lines_out.append(line)
            m = HEADER_START_RE.match(line)
            if m:
                # header_line_index = line_index
                script_lines = []
                state = State.HEADER
                # print("header_start", header_line_index, m)
        elif state == State.HEADER:
            lines_out.append(line)
            m = HEADER_END_RE.match(line)
            if m:
                # content_line_index = line_index
                state = State.CONTENT
                # print("content_start", content_line_index, m)
                script = "".join(script_lines)
                # print(f"script:\n{script}")
                c = compile(script, "<string>", "exec")
                with Capturing() as output:
                    eval(c)
                # print(f"output:\n{output}")
                lines_out += [l + "\n" for l in output]
            else:
                script_lines.append(line)
        elif state == State.CONTENT:
            m = FOOTER_RE.match(line)
            if m:
                lines_out.append(line)
                # footer_line_index = line_index
                state = State.IDLE
                # print("footer", footer_line_index, m)

    print("replaced\n\n")
    print("".join(lines_out))

    open(path, "w").writelines(lines_out)


for path in sys.argv[1:]:
    process_file(Path(path))
