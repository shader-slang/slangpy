# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

PYTHON_SPDX_IDENTIFIER = "# SPDX-License-Identifier: Apache-2.0"
C_SPDX_IDENTIFIER = "// SPDX-License-Identifier: Apache-2.0"


def process_file(file: Path):
    if file.name.endswith(".py"):
        sp = PYTHON_SPDX_IDENTIFIER
    elif file.name.endswith(".slang") or file.name.endswith(".cpp") or file.name.endswith(".c") or file.name.endswith(".h"):
        sp = C_SPDX_IDENTIFIER
    else:
        return

    # read whole file as text
    with open(file, "r") as f:
        text = f.read()

    # split text into lines
    lines = [x for x in text.split("\n") if not x.startswith(sp)]

    lines.insert(0, sp)

    newtext = "\n".join(lines)

    if newtext != text:
        print(file)
        with open(file, "w") as f:
            f.write(newtext)


def run(root: Path):
    files = list(root.glob("**/*.py"))

    for file in files:
        process_file(file)


if __name__ == "__main__":
    run(Path(__file__).parent.parent / "slangpy")
    run(Path(__file__).parent.parent / "tools")
