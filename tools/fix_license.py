# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path

PYTHON_SPDX_IDENTIFIER = "# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception"
C_SPDX_IDENTIFIER = "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception"


def process_file(file: Path):
    if file.name.endswith(".py"):
        sp_start = "# SPDX-License-Identifier:"
        sp = PYTHON_SPDX_IDENTIFIER
    elif file.name.endswith(".slang") or file.name.endswith(".cpp") or file.name.endswith(".c") or file.name.endswith(".h"):
        sp_start = "// SPDX-License-Identifier:"
        sp = C_SPDX_IDENTIFIER
    else:
        return

    # read whole file as text
    with open(file, "r") as f:
        text = f.read()

    # split text into lines
    lines = [x for x in text.split("\n") if not x.startswith(sp_start)]

    lines.insert(0, sp)

    newtext = "\n".join(lines)

    if newtext != text:
        print(file)
        with open(file, "w") as f:
            f.write(newtext)


def run(root: Path):
    files = list(root.glob("**/*.*"))

    for file in files:
        process_file(file)


if __name__ == "__main__":
    run(Path(__file__).parent.parent / "slangpy")
    run(Path(__file__).parent.parent / "tools")
