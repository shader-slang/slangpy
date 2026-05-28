# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#!/usr/bin/env python

"""
A utility for formatting code comments.
"""

import sys
import os
import re
import argparse
from typing import Any, List


DEFAULT_EXTENSIONS = "h,cpp,slang,slangh"


MIN_BLOCK_COMMENT_LINES = 1


def list_files(paths: list[str], recursive: bool = False, extensions: list[str] = []):
    files = []
    for path in paths:
        path = os.path.normpath(path)
        if recursive and os.path.isdir(path):
            for dirpath, _, fnames in os.walk(path):
                fpaths = [os.path.join(dirpath, fname) for fname in fnames]
                for f in fpaths:
                    ext = os.path.splitext(f)[1][1:]
                    if ext in extensions:
                        files.append(f)
        elif os.path.isfile(path):
            files.append(path)
    return files


def prefix_length(s: str):
    m = re.match(r"(\s+)", s)
    if m:
        return len(m.group(0))
    return 0


def create_block_comment(lines: List[str], indent: str, line_ending: str) -> str:
    lines = ["/**", *map(lambda l: " *" if l == "" else " * " + l, lines), " */"]
    return line_ending.join([indent + l for l in lines])


def create_multiline_comment(lines: List[str], indent: str, line_ending: str) -> str:
    return line_ending.join([indent + "/// " + l for l in lines])


def format_comment_line(line: str) -> str:
    # Replace doxygen @xxx with \xxx
    line = re.sub(r"@(\S+)", r"\\\g<1>", line)
    return line


def format_comment(comment: str, is_block_comment: bool) -> str:
    # Determine line ending.
    line_ending = "\r\n" if "\r\n" in comment else "\n"

    indent = prefix_length(comment) * " "
    lines = list(map(lambda l: l.strip(), comment.splitlines()))

    if "/* <<<PYMACRO" in lines[0]:
        return comment

    # Remove leading "///".
    lines = list(map(lambda l: re.sub(r"^\/\/\/\s*", "", l), lines))

    # Check if all lines start with an asterisk.
    all_lines_start_with_asterisk = False
    if is_block_comment:
        all_lines_start_with_asterisk = True
        for index, line in enumerate(lines):
            if index == 0:
                continue
            if index == len(lines) - 1 and "*/" in line:
                continue
            if not (line == "*" or line.startswith("* ")):
                all_lines_start_with_asterisk = False
                break

    # Remove "/(*)+" from the first line.
    lines[0] = re.sub(r"^\/\*+", "", lines[0]).strip()
    # Remove "(*)+/" from the last line.
    lines[-1] = re.sub(r"\*+\/", "", lines[-1]).strip()
    # Remove leading empty lines.
    while (len(lines) > 0) and (lines[0] == ""):
        lines = lines[1:]
    # Remove trailing empty lines.
    while (len(lines) > 0) and (lines[-1] == ""):
        lines = lines[:-1]
    # Remove the asterisk from the beginning of each line.
    if all_lines_start_with_asterisk:
        lines = list(map(lambda l: re.sub(r"^\*[ ]?", "", l), lines))
    # Format each line.
    lines = list(map(format_comment_line, lines))

    if is_block_comment:
        return create_block_comment(lines, indent, line_ending)
    else:
        return create_multiline_comment(lines, indent, line_ending)


def format_comments(text: str) -> str:
    BLOCK_COMMENT = re.compile(r"^[ \t]*\/\*(\*(?!\/)|[^*])*\*\/$", re.DOTALL | re.MULTILINE)
    MULTILINE_COMMENT = re.compile(r"^[ \t]*\/\/\/.*$", re.MULTILINE)

    # Format block comments.
    matches = list(BLOCK_COMMENT.finditer(text))
    for m in reversed(matches):
        comment = m.group()
        new_comment = format_comment(comment, is_block_comment=True)
        # print(f"BLOCK_COMMENT\n{comment}\nNEW:\n{new_comment}\n")
        start = m.start()
        end = m.end()
        text = text[0:start] + new_comment + text[end:]

    # Format multiline comments.
    matches = list(MULTILINE_COMMENT.finditer(text))
    for m in reversed(matches):
        comment = m.group()
        new_comment = format_comment(comment, is_block_comment=False)
        # print(f"BLOCK_COMMENT\n{comment}\nNEW:\n{new_comment}\n")
        start = m.start()
        end = m.end()
        text = text[0:start] + new_comment + text[end:]

    return text


def run(args: Any):
    files = list_files(args.paths, recursive=args.recursive, extensions=args.extensions.split(","))

    for file in files:
        text = open(file, "r").read()
        print(f"Checking file '{file}' ...")
        original = text
        text = format_comments(text)
        if text != original:
            if args.dry_run:
                print(text)
            else:
                print(f"Writing file '{file}'")
                open(file, "w").write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-e",
        "--extensions",
        help="comma separated list of file extensions (default: {})".format(DEFAULT_EXTENSIONS),
        default=DEFAULT_EXTENSIONS,
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="run recursively over directories",
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        default=False,
        help="run without writing files",
    )
    parser.add_argument("paths", metavar="path", nargs="+")

    args = parser.parse_args()

    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
