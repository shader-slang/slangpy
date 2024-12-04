# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def run(root: Path):
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


if __name__ == "__main__":
    run(Path(__file__).parent.parent / "slangpy")
    run(Path(__file__).parent.parent / "tools")
