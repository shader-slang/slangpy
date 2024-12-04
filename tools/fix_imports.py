# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


root = Path(__file__).parent.parent / "slangpy/builtin"
files = list(root.glob("**/*.py"))


def sortfunc(import_line: str):
    import_group = 0
    import_type = 0
    import_priority = 0

    if not "slangpy" in import_line:
        import_group = 0
    else:
        import_group = 1

    if "import" in import_line:
        import_type = 0
    else:
        import_type = 1

    if "core" in import_line:
        import_priority = 0
    else:
        import_priority = 1

    return (import_group, import_type, import_priority, import_line)


for file in files:
    if not file.name.endswith(".py"):
        continue
    if file.name == "__init__.py":
        continue

    # read whole file as text
    with open(file, "r") as f:
        text = f.read()

    # split text into lines
    lines = text.split("\n")

    # split into array that start with "import" and array that doesn't
    imports = []
    rest = []
    found_not_empty_line = False
    for line in lines:
        if not found_not_empty_line and (line.startswith("import") or line.startswith("from")):
            imports.append(line)
        else:
            stripped = line.strip()
            if line.strip() != "" and not line.startswith("#"):
                found_not_empty_line = True
            rest.append(line)

    # sort imports
    imports.sort(key=sortfunc)

    # write back to file
    with open(str(file), "w") as f:
        for line in imports:
            if line.strip() == "":
                continue
            f.write(line + "\n")
        f.write("\n")

        writing = False
        for line in rest:
            if not writing and line.strip() == "":
                continue
            writing = True
            f.write(line + "\n")
