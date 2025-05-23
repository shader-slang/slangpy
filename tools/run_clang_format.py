#!/usr/bin/env python

# https://github.com/Sarcasm/run-clang-format

# MIT License
#
# Copyright (c) 2017 Guillaume Papin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This script has been extended with the ability to format slang files.

"""A wrapper script around clang-format, suitable for linting multiple files
and to use for continuous integration.

This is an alternative API for the clang-format command line.
It runs over multiple files and directories in parallel.
A diff output is produced and a sensible exit code is returned.

"""

from __future__ import print_function, unicode_literals

import argparse
import difflib
import fnmatch
import io
import errno
import multiprocessing
import os
import signal
import subprocess
import sys
import traceback
from typing import Any, Optional, Sequence
import xml.etree.ElementTree as ET
from pathlib import Path

from functools import partial

try:
    from subprocess import DEVNULL  # py3k
except ImportError:
    DEVNULL = open(os.devnull, "wb")  # type: ignore (declared final)


DEFAULT_EXTENSIONS = "h,cpp,slang,slangh"
DEFAULT_SLANG_EXTENSIONS = "slang,slangh"
DEFAULT_CLANG_FORMAT_IGNORE = ".clang-format-ignore"


class ExitStatus:
    SUCCESS = 0
    DIFF = 1
    TROUBLE = 2


def excludes_from_file(ignore_file: str):
    excludes = []
    whitelist = []
    try:
        with io.open(ignore_file, "r", encoding="utf-8") as f:
            for line in f:
                pattern = line.strip()
                if line.startswith("#"):
                    # ignore comments
                    continue
                is_whitelist = line.startswith("!")
                if is_whitelist:
                    pattern = pattern[1:]
                if not pattern:
                    # allow empty lines
                    continue
                if is_whitelist:
                    whitelist.append(pattern)
                else:
                    excludes.append(pattern)
    except EnvironmentError as e:
        if e.errno != errno.ENOENT:
            raise
    return [excludes, whitelist]


def is_child(path: str, files: list[str]):
    if path in files:
        return True
    test_path = Path(path)
    for file in files:
        if Path(file) in test_path.parents:
            return True
    return False


def list_files(
    files: list[str],
    recursive: bool = False,
    extensions: Optional[list[str]] = None,
    exclude: Optional[list[str]] = None,
    whitelist: Optional[list[str]] = None,
):
    if extensions is None:
        extensions = []
    if exclude is None:
        exclude = []
    if whitelist is None:
        whitelist = []

    out = set()
    for file in files:
        if recursive and os.path.isdir(file):
            for dirpath, dnames, fnames in os.walk(file):
                fpaths = [os.path.join(dirpath, fname) for fname in fnames]
                for pattern in exclude:
                    # os.walk() supports trimming down the dnames list
                    # by modifying it in-place,
                    # to avoid unnecessary directory listings.
                    dnames[:] = [
                        x
                        for x in dnames
                        if not fnmatch.fnmatch(os.path.join(dirpath, x), pattern)
                    ]
                    fpaths = [x for x in fpaths if not fnmatch.fnmatch(x, pattern)]
                for f in fpaths:
                    ext = os.path.splitext(f)[1][1:]
                    if ext in extensions:
                        out.add(f)
        else:
            out.add(file)

    if not recursive:
        return out

    # process whitelist, whole whitelisted directories, and also specific files
    for path in whitelist:
        if is_child(path, files):
            if os.path.isfile(path):
                ext = os.path.splitext(path)[1][1:]
                if ext in extensions:
                    out.add(path)
            if os.path.isdir(path):
                subsearch = list_files([path], True, extensions)
                out = out.union(subsearch)

    # normalize paths
    out = [os.path.normpath(p) for p in out]

    return out


def make_diff(file: str, original: list[str], reformatted: list[str]):
    return list(
        difflib.unified_diff(
            original,
            reformatted,
            fromfile="{}\t(original)".format(file),
            tofile="{}\t(reformatted)".format(file),
            n=3,
        )
    )


class DiffError(Exception):
    def __init__(self, message: str, errs: Optional[list[str]] = None):
        super(DiffError, self).__init__(message)
        self.errs = errs or []


class UnexpectedError(Exception):
    def __init__(self, message: str, exc: Optional[Exception] = None):
        super(UnexpectedError, self).__init__(message)
        self.formatted_traceback = traceback.format_exc()
        self.exc = exc


def run_clang_format_diff_wrapper(args: Any, file: str):
    try:
        ret = run_clang_format_diff(args, file)
        return ret
    except DiffError:
        raise
    except Exception as e:
        raise UnexpectedError("{}: {}: {}".format(file, e.__class__.__name__, e), e)


def run_clang_format_diff(args: Any, file: str) -> tuple[list[str], list[str]]:
    ext = os.path.splitext(file)[1][1:]
    is_slang = ext in args.slang_extensions.split(",")

    try:
        ins = open(file, "rb").read()
    except IOError as exc:
        raise DiffError(str(exc))

    invocation = [args.clang_format_executable]

    if args.style:
        invocation.extend(["--style", args.style])

    # If this is a slang file, we process it as C# and do the replacements manually so we can filter unwanted ones
    if is_slang:
        invocation.extend(["--assume-filename", "source.cs"])
        invocation.extend(["--output-replacements-xml"])

    invocation.extend(["--"])

    if args.dry_run:
        print(" ".join(invocation))
        return [], []

    try:
        proc = subprocess.Popen(
            invocation,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise DiffError(
            "Command '{}' failed to start: {}".format(
                subprocess.list2cmdline(invocation), exc
            )
        )

    outs, errs = proc.communicate(input=ins)

    # Apply replacements
    if is_slang:
        if outs != "":
            replacements = ET.fromstring(outs)
            formatted = ins
            for replacement in reversed(replacements):
                offset = int(replacement.attrib["offset"])
                length = int(replacement.attrib["length"])
                text = replacement.text
                if text == None:
                    text = ""

                # Do not allow inserting a new line after '}' and before ';'
                if (
                    length == 0
                    and formatted[offset] == ord(";")
                    and formatted[offset - 1] == ord("}")
                ):
                    continue

                formatted = (
                    formatted[:offset]
                    + text.encode("utf-8")
                    + formatted[offset + length :]
                )

            outs = formatted
        else:
            outs = ins

    original_lines = ins.decode("utf-8").splitlines(keepends=True)
    formatted_lines = outs.decode("utf-8").splitlines(keepends=True)
    decoded_errs = errs.decode("utf-8").splitlines(keepends=True)

    if proc.returncode:
        raise DiffError(
            "Command '{}' returned non-zero exit status {}".format(
                subprocess.list2cmdline(invocation), proc.returncode
            ),
            decoded_errs,
        )

    if args.in_place and outs != ins:
        open(file, "wb").write(outs)
        return [], decoded_errs

    return make_diff(file, original_lines, formatted_lines), decoded_errs


def bold_red(s: str):
    return "\x1b[1m\x1b[31m" + s + "\x1b[0m"


def colorize(diff_lines: Sequence[str]):
    def bold(s: str):
        return "\x1b[1m" + s + "\x1b[0m"

    def cyan(s: str):
        return "\x1b[36m" + s + "\x1b[0m"

    def green(s: str):
        return "\x1b[32m" + s + "\x1b[0m"

    def red(s: str):
        return "\x1b[31m" + s + "\x1b[0m"

    for line in diff_lines:
        if line[:4] in ["--- ", "+++ "]:
            yield bold(line)
        elif line.startswith("@@ "):
            yield cyan(line)
        elif line.startswith("+"):
            yield green(line)
        elif line.startswith("-"):
            yield red(line)
        else:
            yield line


def print_diff(diff_lines: Sequence[str], use_color: bool):
    if use_color:
        diff_lines = list(colorize(diff_lines))
    if sys.version_info[0] < 3:
        sys.stdout.writelines((l.encode("utf-8") for l in diff_lines))
    else:
        sys.stdout.writelines(diff_lines)


def print_trouble(prog: str, message: str, use_colors: bool):
    error_text = "error:"
    if use_colors:
        error_text = bold_red(error_text)
    print("{}: {} {}".format(prog, error_text, message), file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clang-format-executable",
        metavar="EXECUTABLE",
        help="path to the clang-format executable",
        default="clang-format",
    )
    parser.add_argument(
        "--extensions",
        help="comma separated list of file extensions (default: {})".format(
            DEFAULT_EXTENSIONS
        ),
        default=DEFAULT_EXTENSIONS,
    )
    parser.add_argument(
        "--slang-extensions",
        help="comma separated list of slang file extensions (default: {})".format(
            DEFAULT_SLANG_EXTENSIONS
        ),
        default=DEFAULT_SLANG_EXTENSIONS,
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="run recursively over directories",
    )
    parser.add_argument(
        "-d", "--dry-run", action="store_true", help="just print the list of files"
    )
    parser.add_argument(
        "-i",
        "--in-place",
        action="store_true",
        help="format file instead of printing differences",
    )
    parser.add_argument("files", metavar="file", nargs="+")
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="disable output, useful for the exit code",
    )
    parser.add_argument(
        "-j",
        metavar="N",
        type=int,
        default=0,
        help="run N clang-format jobs in parallel" " (default number of cpus + 1)",
    )
    parser.add_argument(
        "--color",
        default="auto",
        choices=["auto", "always", "never"],
        help="show colored diff (default: auto)",
    )
    parser.add_argument(
        "-e",
        "--exclude",
        metavar="PATTERN",
        action="append",
        default=[],
        help="exclude paths matching the given glob-like pattern(s)"
        " from recursive search",
    )
    parser.add_argument(
        "--style",
        help="formatting style to apply (LLVM, Google, Chromium, Mozilla, WebKit)",
    )

    args = parser.parse_args()

    # use default signal handling, like diff return SIGINT value on ^C
    # https://bugs.python.org/issue14229#msg156446
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    try:
        signal.SIGPIPE  # type: ignore (windows)
    except AttributeError:
        # compatibility, SIGPIPE does not exist on Windows
        pass
    else:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # type: ignore (windows)

    colored_stdout = False
    colored_stderr = False
    if args.color == "always":
        colored_stdout = True
        colored_stderr = True
    elif args.color == "auto":
        colored_stdout = sys.stdout.isatty()
        colored_stderr = sys.stderr.isatty()

    version_invocation = [args.clang_format_executable, str("--version")]
    try:
        subprocess.check_call(version_invocation, stdout=DEVNULL)
    except subprocess.CalledProcessError as e:
        print_trouble(parser.prog, str(e), use_colors=colored_stderr)
        return ExitStatus.TROUBLE
    except OSError as e:
        print_trouble(
            parser.prog,
            "Command '{}' failed to start: {}".format(
                subprocess.list2cmdline(version_invocation), e
            ),
            use_colors=colored_stderr,
        )
        return ExitStatus.TROUBLE

    retcode = ExitStatus.SUCCESS

    [excludes, whitelist] = excludes_from_file(DEFAULT_CLANG_FORMAT_IGNORE)
    excludes.extend(args.exclude)

    files = list_files(
        args.files,
        recursive=args.recursive,
        exclude=excludes,
        extensions=args.extensions.split(","),
        whitelist=whitelist,
    )

    if not files:
        return

    njobs = args.j
    if njobs == 0:
        njobs = multiprocessing.cpu_count() + 1
    # fix issue on windows
    if sys.platform == "win32":
        njobs = min(60, njobs)
    njobs = min(len(files), njobs)

    if njobs == 1:
        # execute directly instead of in a pool,
        # less overhead, simpler stacktraces
        it = (run_clang_format_diff_wrapper(args, file) for file in files)
        pool = None
    else:
        pool = multiprocessing.Pool(njobs)
        it = pool.imap_unordered(partial(run_clang_format_diff_wrapper, args), files)
        pool.close()
    while True:
        try:
            outs, errs = next(it)
        except StopIteration:
            break
        except DiffError as e:
            print_trouble(parser.prog, str(e), use_colors=colored_stderr)
            retcode = ExitStatus.TROUBLE
            sys.stderr.writelines(e.errs)
        except UnexpectedError as e:
            print_trouble(parser.prog, str(e), use_colors=colored_stderr)
            sys.stderr.write(e.formatted_traceback)
            retcode = ExitStatus.TROUBLE
            # stop at the first unexpected error,
            # something could be very wrong,
            # don't process all files unnecessarily
            if pool:
                pool.terminate()
            break
        else:
            sys.stderr.writelines(errs)
            if outs == []:
                continue
            if not args.quiet:
                print_diff(outs, use_color=colored_stdout)
            if retcode == ExitStatus.SUCCESS:
                retcode = ExitStatus.DIFF
    if pool:
        pool.join()

    return retcode


if __name__ == "__main__":
    sys.exit(main())
