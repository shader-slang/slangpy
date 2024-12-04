
from pathlib import Path

SPDX_IDENTIFIER = "SPDX-License-Identifier: Apache-2.0"


root = Path(__file__).parent.parent / "slangpy"
files = list(root.glob("**/*.py"))


for file in files:
    if not file.name.endswith(".py"):
        continue

    # read whole file as text
    with open(file, "r") as f:
        text = f.read()

    # split text into lines
    lines = [x for x in text.split("\n") if not SPDX_IDENTIFIER in x]

    lines.insert(0, f"# {SPDX_IDENTIFIER}")

    newtext = "\n".join(lines)

    if newtext != text:
        print(file)
        with open(file, "w") as f:
            f.write(newtext)
