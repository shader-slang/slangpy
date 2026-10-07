"""
This is a package for building pybind11 docstrings from C++ header comments.

(Docs WIP).
"""

import argparse
import os
import sys
from pathlib import Path

from pybind11_mkdoc.mkdoc_lib import mkdoc

__version__ = "3.0.0"


def _append_include_dir(
    args: list[str], include_dir: str, *, verbose: bool = True
) -> None:
    """
    Add an include directory to an argument list (if it exists).

    Parameters
    ----------

    args: list
        The list to append the include directory to.

    include_dir: str
        The include directory to append.

    verbose: bool
        Whether to print a warning for non-existing directories.
    """

    if os.path.isdir(include_dir):
        args.append(f"-I{include_dir}")
    elif verbose:
        print(f"Include directory '{include_dir}' does not exist!", file=sys.stderr)


def _append_definition(args: list[str], definition: str) -> None:
    """
    Add a compiler definition to an argument list.

    The definition is expected to be given in the format '<macro>=<value>',
    which will define <macro> to <value> (or 1 if the '=<value>' part is
    omitted). An explicit empty value ('<macro>=') defines <macro> to nothing.

    Parameters
    ----------

    args: list
        The list to append the definition to.

    definition: str
        The definition to append.
    """

    macro, sep, value = definition.partition("=")
    value = value.strip() if sep else "1"

    args.append(f"-D{macro.strip()}={value}")


def get_cmake_dir() -> Path:
    """
    Return the path to the pybind11_mkdoc CMake module directory.
    """
    cmake_installed_path = Path(__file__).parent / "share" / "cmake" / "pybind11_mkdoc"
    if cmake_installed_path.exists():
        return cmake_installed_path

    msg = "pybind11_mkdoc cmake files not found."
    raise ImportError(msg)


def main() -> int:
    """
    Entry point for the `pybind11_mkdoc` console script.

    Parses the  commandline arguments given to the console script and passes them on to `mkdoc`.
    """

    parser = argparse.ArgumentParser(
        prog="pybind11_mkdoc",
        description="Processes a sequence of C/C++ headers and extracts comments for use in pybind11 binding code.",
        epilog="(Other compiler flags that Clang understands can also be supplied)",
        allow_abbrev=False,
    )

    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )

    parser.add_argument(
        "-o",
        "--output",
        action="store",
        type=str,
        dest="output",
        metavar="<file>",
        help="Write to the specified file (default: use stdout).",
    )

    parser.add_argument(
        "-w",
        "--width",
        action="store",
        type=int,
        dest="width",
        metavar="<width>",
        help="Specify docstring width before wrapping.",
    )

    parser.add_argument(
        "-I",
        action="append",
        type=str,
        dest="include_dirs",
        metavar="<dir>",
        help="Specify an directory to add to the list of include search paths.",
    )

    parser.add_argument(
        "-D",
        action="append",
        type=str,
        metavar="<macro>=<value>",
        dest="definitions",
        help="Specify a compiler definition, i.e. define <macro> to <value> (or 1 if <value> omitted).",
    )

    parser.add_argument("header", type=str, nargs="+", help="A header file to process.")

    parsed_args, unparsed_args = parser.parse_known_args()

    mkdoc_args: list[str] = []
    mkdoc_out = parsed_args.output
    docstring_width = parsed_args.width

    if parsed_args.include_dirs is not None:
        for include_dir in parsed_args.include_dirs:
            _append_include_dir(mkdoc_args, include_dir)

    if parsed_args.definitions is not None:
        for definition in parsed_args.definitions:
            _append_definition(mkdoc_args, definition)

    for arg in unparsed_args:
        if arg.startswith("-I"):
            _append_include_dir(mkdoc_args, arg[2:])
        elif arg.startswith("-D"):
            _append_definition(mkdoc_args, arg[2:])
        else:
            # append argument as is and hope for the best
            mkdoc_args.append(arg)

    mkdoc_args.extend(parsed_args.header)

    mkdoc(mkdoc_args, docstring_width, mkdoc_out)

    return 0
