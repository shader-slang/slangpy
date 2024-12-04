# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

SPDX_IDENTIFIER = "# SPDX-License-Identifier: Apache-2.0"


def process_file(file: Path):
    if not file.name.endswith(".py"):
        return

    # read whole file as text
    with open(file, "r") as f:
        text = f.read()

    # split text into lines
    lines = [x for x in text.split("\n") if not x.startswith(SPDX_IDENTIFIER)]

    lines.insert(0, SPDX_IDENTIFIER)

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
