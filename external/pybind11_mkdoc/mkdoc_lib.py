#!/usr/bin/env python3
#
#  Syntax: mkdoc.py [-I<path> ..] [.. a list of header files ..]
#
#  Extract documentation from C++ header files to use it in Python bindings
#

from __future__ import annotations

import contextlib
import ctypes.util
import os
import platform
import re
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from itertools import repeat
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, TextIO, Union

from clang import cindex
from clang.cindex import CursorKind

if TYPE_CHECKING:
    from collections.abc import Iterable

# (name, filename, comment) for one documented declaration
_Doc = tuple[str, str, str]

# The active continuation target while consuming Doxygen sections: a tag
# plus the container (and index) that follow-up lines are appended to.
_Target = Union[
    tuple[Literal["body", "brief"], "list[str]"],
    tuple[Literal["section"], "list[tuple[str, list[str]]]"],
    tuple[Literal["list"], "list[str]", int],
    tuple[Literal["entry"], "list[tuple[str, str]]", int],
]

RECURSE_LIST = [
    CursorKind.TRANSLATION_UNIT,
    CursorKind.NAMESPACE,
    CursorKind.CLASS_DECL,
    CursorKind.STRUCT_DECL,
    CursorKind.ENUM_DECL,
    CursorKind.CLASS_TEMPLATE,
]

PRINT_LIST = [
    CursorKind.CLASS_DECL,
    CursorKind.STRUCT_DECL,
    CursorKind.ENUM_DECL,
    CursorKind.ENUM_CONSTANT_DECL,
    CursorKind.CLASS_TEMPLATE,
    CursorKind.FUNCTION_DECL,
    CursorKind.FUNCTION_TEMPLATE,
    CursorKind.CONVERSION_FUNCTION,
    CursorKind.CXX_METHOD,
    CursorKind.CONSTRUCTOR,
    CursorKind.FIELD_DECL,
]

FUNCTION_DOCSTRING_LIST = [
    CursorKind.FUNCTION_DECL,
    CursorKind.FUNCTION_TEMPLATE,
    CursorKind.CONVERSION_FUNCTION,
    CursorKind.CXX_METHOD,
    CursorKind.CONSTRUCTOR,
]

PREFIX_BLACKLIST = [CursorKind.TRANSLATION_UNIT]

CPP_OPERATORS = {
    "<=": "le",
    ">=": "ge",
    "==": "eq",
    "!=": "ne",
    "[]": "array",
    "+=": "iadd",
    "-=": "isub",
    "*=": "imul",
    "/=": "idiv",
    "%=": "imod",
    "&=": "iand",
    "|=": "ior",
    "^=": "ixor",
    "<<=": "ilshift",
    ">>=": "irshift",
    "++": "inc",
    "--": "dec",
    "<<": "lshift",
    ">>": "rshift",
    "&&": "land",
    "||": "lor",
    "!": "lnot",
    "~": "bnot",
    "&": "band",
    "|": "bor",
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "div",
    "%": "mod",
    "<": "lt",
    ">": "gt",
    "=": "assign",
    "()": "call",
}

# Longest operators first, so that e.g. "<<=" is matched before "<<".
CPP_OPERATORS = dict(sorted(CPP_OPERATORS.items(), key=lambda t: -len(t[0])))

docstring_width = 70


class NoFilenamesError(ValueError):
    pass


def sanitize_name(name: str) -> str:
    name = re.sub(r"type-parameter-0-([0-9]+)", r"T\1", name)
    for k, v in CPP_OPERATORS.items():
        name = name.replace(f"operator{k}", f"operator_{v}")
    name = re.sub("<.*>", "", name)
    name = "".join([ch if ch.isalnum() else "_" for ch in name])
    name = re.sub("_+", "_", name).removesuffix("_")
    return "mkd_doc_" + name


section_command_re = re.compile(r"\s*[\\@](\w+)(?:\[([^\]]+)\])?(?:\s+(.*))?$")
code_segment_re = re.compile(r"(```)")
prefix_re = re.compile(
    r"(\s*)((?:[*\-•]\s)|(?:\(?\d+[\.)]\s)|(?:[\w:]+(?:\s+\[[^\]]+\])?:(?:\s+|$)))"
)
named_arg_re = re.compile(r"([\w:]+)\s*(.*)")

IGNORED_DOXYGEN_COMMANDS = {
    "addtogroup",
    "class",
    "def",
    "defgroup",
    "dir",
    "enum",
    "file",
    "fn",
    "ingroup",
    "interface",
    "mainpage",
    "name",
    "namespace",
    "overload",
    "package",
    "page",
    "private",
    "protected",
    "public",
    "relates",
    "relatesalso",
    "struct",
    "typedef",
    "union",
    "var",
    "weakgroup",
}

SECTION_HEADINGS = {
    "attention": "Attention",
    "author": "Author",
    "authors": "Authors",
    "bug": "Bug",
    "copyright": "Copyright",
    "date": "Date",
    "deprecated": "Deprecated",
    "invariant": "Invariant",
    "note": "Note",
    "par": None,
    "post": "Postcondition",
    "pre": "Precondition",
    "remark": "Remark",
    "remarks": "Remarks",
    "result": "Returns",
    "retval": "Returns",
    "return": "Returns",
    "returns": "Returns",
    "sa": "See also",
    "see": "See also",
    "since": "Since",
    "todo": "Todo",
    "version": "Version",
    "warning": "Warning",
}

INLINE_DOXYGEN_REPLACEMENTS = [
    (re.compile(r"[\\@][cp]\s+([^\s]+)"), r"``\1``"),
    (re.compile(r"[\\@](?:a|e|em)\s+([^\s]+)"), r"*\1*"),
    (re.compile(r"[\\@]b\s+([^\s]+)"), r"**\1**"),
    (re.compile(r"[\\@]ref\s+([^\s]+)"), r"\1"),
]

# Applied before the Doxygen sections are consumed.
BLOCK_DOXYGEN_REPLACEMENTS = [
    (
        re.compile(r"[\\@]code\s?(.*?)\s?[\\@]endcode", flags=re.DOTALL),
        "```\n\\1\n```\n",
    ),
    (
        re.compile(r"[\\@]verbatim\s?(.*?)\s?[\\@]endverbatim", flags=re.DOTALL),
        "```\n\\1\n```\n",
    ),
]

MARKUP_REPLACEMENTS = [
    (re.compile(r"<tt>(.*?)</tt>", flags=re.DOTALL), r"``\1``"),
    (re.compile(r"<pre>(.*?)</pre>", flags=re.DOTALL), "```\n\\1\n```\n"),
    (re.compile(r"<em>(.*?)</em>", flags=re.DOTALL), r"*\1*"),
    (re.compile(r"<b>(.*?)</b>", flags=re.DOTALL), r"**\1**"),
    (re.compile(r"[\\@]f\$(.*?)[\\@]f\$", flags=re.DOTALL), r":math:`\1`"),
    (re.compile(r"<li>"), "\n\n* "),
    (re.compile(r"</?ul>"), ""),
    (re.compile(r"</li>"), "\n\n"),
]


def _format_named_entries(heading: str, entries: list[tuple[str, str]]) -> list[str]:
    if not entries:
        return []
    lines = [f"{heading}:"]
    lines += [f"    {name}: {text}" for name, text in entries]
    lines.append("")
    return lines


def _format_list_entries(heading: str, entries: list[str]) -> list[str]:
    if not entries:
        return []
    lines = [f"{heading}:"]
    lines += [f"    {text}" for text in entries]
    lines.append("")
    return lines


def _format_section_entries(entries: Iterable[tuple[str, str]]) -> list[str]:
    lines = []
    for heading, text in entries:
        lines.append(f"{heading}:")
        if text:
            lines.append(f"    {text}")
        lines.append("")
    return lines


def _append_continuation(target: _Target, text: str) -> None:
    text = text.strip()
    if not text:
        return
    if target[0] == "body":
        target[1].append(text)
    elif target[0] == "brief":
        if target[1] and target[1][-1].rstrip().endswith((".", "!", "?")):
            target[1].append("")
        target[1].append(text)
    elif target[0] == "section":
        target[1][-1][1].append(text)
    elif target[0] == "list":
        entries = target[1]
        index = target[2]
        entries[index] = f"{entries[index]} {text}".strip()
    else:
        named_entries = target[1]
        index = target[2]
        name, value = named_entries[index]
        named_entries[index] = (name, f"{value} {text}".strip())


def _consume_doxygen_sections(s: str) -> str:
    lines = s.splitlines()
    body_lines: list[str] = []
    params: list[tuple[str, str]] = []
    t_params: list[tuple[str, str]] = []
    returns: list[str] = []
    raises: list[tuple[str, str]] = []
    sections: list[tuple[str, list[str]]] = []
    active: _Target | None = None
    pending_brief_separator = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            body_lines.append(line)
            active = None
            pending_brief_separator = False
            continue

        m = section_command_re.match(line)
        if not m:
            if active is not None:
                _append_continuation(active, line)
            else:
                if pending_brief_separator:
                    body_lines.append("")
                    pending_brief_separator = False
                body_lines.append(line)
            continue

        command, option, rest = m.groups()
        command = command.lower()
        rest = (rest or "").strip()

        if pending_brief_separator and command in {"details", "short"}:
            body_lines.append("")
            pending_brief_separator = False

        if command in {"brief", "short"}:
            if rest:
                body_lines.append(rest)
                pending_brief_separator = True
                active = ("brief", body_lines)
            else:
                active = None
            continue

        if command == "details":
            if rest:
                body_lines.append(rest)
                active = ("body", body_lines)
            else:
                active = None
            continue

        if command in {"param", "arg"}:
            arg = named_arg_re.match(rest)
            if arg:
                name, text = arg.groups()
                if option:
                    name = f"{name} [{option}]"
                params.append((name, text.strip()))
                active = ("entry", params, len(params) - 1)
            else:
                active = None
            continue

        if command in {"tparam", "typeparam"}:
            arg = named_arg_re.match(rest)
            if arg:
                name, text = arg.groups()
                t_params.append((name, text.strip()))
                active = ("entry", t_params, len(t_params) - 1)
            else:
                active = None
            continue

        if command in {"return", "returns", "result"}:
            if rest:
                returns.append(rest)
                active = ("list", returns, len(returns) - 1)
            else:
                active = None
            continue

        if command == "retval":
            arg = named_arg_re.match(rest)
            if arg:
                name, text = arg.groups()
                returns.append(f"{name}: {text.strip()}" if text else name)
                active = ("list", returns, len(returns) - 1)
            else:
                active = None
            continue

        if command in {"exception", "throw", "throws"}:
            arg = named_arg_re.match(rest)
            if arg:
                name, text = arg.groups()
                raises.append((name, text.strip()))
                active = ("entry", raises, len(raises) - 1)
            else:
                active = None
            continue

        if command in IGNORED_DOXYGEN_COMMANDS:
            active = None
            continue

        if command in SECTION_HEADINGS:
            heading = SECTION_HEADINGS[command]
            if heading is None:
                heading = rest.strip().rstrip(":") or "Note"
                rest = ""
            sections.append((heading, [rest] if rest else []))
            active = ("section", sections)
            continue

        if active is not None:
            _append_continuation(active, line)
        else:
            body_lines.append(line)

    result = list(body_lines)
    while result and not result[-1].strip():
        result.pop()
    if result and (params or t_params or returns or raises or sections):
        result.append("")
    result += _format_named_entries("Args", params)
    result += _format_named_entries("Template Args", t_params)
    result += _format_list_entries("Returns", returns)
    result += _format_named_entries("Raises", raises)
    result += _format_section_entries(
        (heading, " ".join(text).strip()) for heading, text in sections
    )
    return "\n".join(result)


def process_comment(comment: str) -> str:
    # For declarations that have no raw comment; skip the full normalization pipeline.
    if not comment:
        return ""

    result_lines = []

    # Remove C++ comment syntax
    leading_spaces: int | None = None
    for s in comment.expandtabs(tabsize=4).splitlines():
        s = s.strip()
        if s.endswith("*/"):
            s = s[:-2].rstrip("*")
        if s.startswith("/*"):
            s = s[2:].lstrip("*!<")
        elif s.startswith("//"):
            s = s[2:].lstrip("/!<")
        elif s.startswith("*"):
            s = s[1:]
        if len(s) > 0:
            n_spaces = len(s) - len(s.lstrip())
            leading_spaces = (
                n_spaces if leading_spaces is None else min(leading_spaces, n_spaces)
            )
        result_lines.append(s)

    if leading_spaces:
        s = "\n".join(s[leading_spaces:] for s in result_lines) + "\n"
    else:
        s = "\n".join(result_lines) + "\n"
    for pattern, replacement in INLINE_DOXYGEN_REPLACEMENTS:
        s = pattern.sub(replacement, s)

    for pattern, replacement in BLOCK_DOXYGEN_REPLACEMENTS:
        s = pattern.sub(replacement, s)

    s = _consume_doxygen_sections(s)

    # HTML/TeX tags
    for pattern, replacement in MARKUP_REPLACEMENTS:
        s = pattern.sub(replacement, s)

    s = s.replace("``true``", "``True``")
    s = s.replace("``false``", "``False``")

    # Re-flow text
    wrapper = textwrap.TextWrapper()
    wrapper.expand_tabs = True
    wrapper.replace_whitespace = True
    wrapper.drop_whitespace = True
    wrapper.width = docstring_width
    wrapper.initial_indent = wrapper.subsequent_indent = ""

    parts: list[str] = []
    in_code_segment = False
    for x in code_segment_re.split(s):
        if x == "```":
            if not in_code_segment:
                if parts and parts[-1] and not parts[-1].endswith("\n\n"):
                    parts[-1] += "\n"
                parts.append("```\n")
            else:
                parts.append("\n```\n\n")
            in_code_segment = not in_code_segment
        elif in_code_segment:
            parts.append(x.strip())
        else:
            wrapped: list[str] = []
            paragraph: list[str] = []

            def get_prefix_and_indent(line: str) -> tuple[str | None, str]:
                indent = len(line) - len(line.lstrip())
                indent_str = " " * indent
                m = prefix_re.match(line)
                if m:
                    prefix = m.group(0)
                    return prefix, " " * len(prefix)
                return None, indent_str

            def flush_paragraph(
                paragraph: list[str] = paragraph, wrapped: list[str] = wrapped
            ) -> None:
                if not paragraph:
                    return

                # Detect bullet/number from first line
                first_line = paragraph[0]
                prefix, indent_str = get_prefix_and_indent(first_line)

                # Combine paragraph into single string (replace internal line breaks with space)
                para_text = " ".join(line.strip() for line in paragraph)

                if prefix:
                    content = para_text[len(prefix.lstrip()) :]
                    wrapper.initial_indent = prefix
                    wrapper.subsequent_indent = indent_str
                    if content == "":
                        # This paragraph is just the prefix
                        wrapped.append(prefix)
                        paragraph.clear()
                        return
                else:
                    content = para_text.lstrip()
                    wrapper.initial_indent = indent_str
                    wrapper.subsequent_indent = indent_str

                wrapped.append(wrapper.fill(content))
                paragraph.clear()

            current_prefix: str | None = None
            current_indent = ""
            for line in x.splitlines():
                if not line.strip():
                    flush_paragraph()
                    wrapped.append(line)  # preserve blank lines
                    continue

                prefix, indent = get_prefix_and_indent(line)
                if paragraph and (
                    (indent != current_indent) or (prefix and prefix != current_prefix)
                ):
                    # Prefix/indent changed → start new paragraph
                    flush_paragraph()

                paragraph.append(line)
                current_prefix = prefix
                current_indent = indent

            flush_paragraph()
            parts.append("\n".join(wrapped))
    return "".join(parts).rstrip().lstrip("\n")


def format_function_docstring(comment: str) -> str:
    if not comment:
        return ""
    comment = comment.rstrip()
    if "\n" not in comment:
        return comment
    return comment + "\n\n"


def _is_cursor_from_file(node: Any, filename: str, file_cache: dict[str, bool]) -> bool:
    if node.location.file is None:
        return True

    node_filename = node.location.file.name
    # libclang often reports many cursors from the same file; avoid repeated stat calls by caching.
    if node_filename not in file_cache:
        file_cache[node_filename] = os.path.samefile(node_filename, filename)
    return file_cache[node_filename]


def extract(
    filename: str,
    node: Any,
    prefix: str,
    output: list[_Doc],
    file_cache: dict[str, bool],
) -> None:
    if not _is_cursor_from_file(node, filename, file_cache):
        return
    if node.kind in RECURSE_LIST:
        sub_prefix = prefix
        if node.kind not in PREFIX_BLACKLIST:
            if len(sub_prefix) > 0:
                sub_prefix += "_"
            sub_prefix += node.spelling
        for i in node.get_children():
            extract(filename, i, sub_prefix, output, file_cache)
    if node.kind in PRINT_LIST:
        comment = process_comment(node.raw_comment or "")
        if node.kind in FUNCTION_DOCSTRING_LIST:
            comment = format_function_docstring(comment)
        sub_prefix = prefix
        if len(sub_prefix) > 0:
            sub_prefix += "_"
        if len(node.spelling) > 0:
            name = sanitize_name(sub_prefix + node.spelling)
            output.append((name, filename, comment))


def _report_diagnostics(filename: str, tu: Any) -> None:
    """Print clang diagnostics to stderr and fail if any of them is an error."""
    errors = 0
    for diagnostic in tu.diagnostics:
        sys.stderr.write(diagnostic.format() + "\n")
        if diagnostic.severity >= cindex.Diagnostic.Error:
            errors += 1
    if errors:
        msg = f"Clang reported {errors} error(s) while parsing {filename}"
        raise RuntimeError(msg)


def _extract_file(filename: str, parameters: list[str]) -> list[_Doc]:
    # Diagnostics are printed by _report_diagnostics, not by libclang itself.
    index = cindex.Index(cindex.conf.lib.clang_createIndex(False, False))
    tu = index.parse(filename, parameters)
    _report_diagnostics(filename, tu)
    output: list[_Doc] = []
    extract(filename, tu.cursor, "", output, {})
    return output


def _folder_version(d: str) -> list[int]:
    return [int(ver) for ver in re.findall(r"(?<!lib)(?<!\d)\d+", d)]


def read_args(args: list[str]) -> tuple[list[str], list[str]]:
    parameters = []
    filenames = []
    if "-x" not in args:
        parameters.extend(["-x", "c++"])
    if not any(it.startswith("-std=") for it in args):
        parameters.append("-std=c++17")
    parameters.append("-Wno-pragma-once-outside-header")

    if platform.system() == "Darwin":
        dev_path = "/Applications/Xcode.app/Contents/Developer/"
        clt_path = "/Library/Developer/CommandLineTools/"

        # cindex forbids (re)configuring the library once it has been loaded
        if "LIBCLANG_PATH" in os.environ:
            library_file = os.environ["LIBCLANG_PATH"]
            if not os.path.isfile(library_file):
                msg = (
                    f"LIBCLANG_PATH points to {library_file!r}, which is not a file. "
                    "Set it to the path of libclang.dylib."
                )
                raise FileNotFoundError(msg)
            if not cindex.Config.loaded:
                cindex.Config.set_library_file(library_file)
        else:
            for libclang in (
                dev_path + "Toolchains/XcodeDefault.xctoolchain/usr/lib/libclang.dylib",
                clt_path + "usr/lib/libclang.dylib",
            ):
                if os.path.exists(libclang):
                    if not cindex.Config.loaded:
                        cindex.Config.set_library_path(os.path.dirname(libclang))
                    break

        for sdk_dir in (
            dev_path + "Platforms/MacOSX.platform/Developer/SDKs",
            clt_path + "SDKs",
        ):
            if os.path.exists(sdk_dir):
                sdks = sorted(next(os.walk(sdk_dir))[1], key=_folder_version)
                if "MacOSX.sdk" in sdks:
                    sdk = "MacOSX.sdk"
                elif sdks:
                    sdk = sdks[-1]
                else:
                    continue
                parameters.extend(["-isysroot", str(PurePosixPath(sdk_dir) / sdk)])
                break
    elif platform.system() == "Windows":
        if "LIBCLANG_PATH" in os.environ:
            library_file = os.environ["LIBCLANG_PATH"]
            if not os.path.isfile(library_file):
                msg = (
                    f"LIBCLANG_PATH points to {library_file!r}, which is not a file. "
                    "Set it to the path of libclang.dll."
                )
                raise FileNotFoundError(msg)
            if not cindex.Config.loaded:
                cindex.Config.set_library_file(library_file)
        else:
            found_library = ctypes.util.find_library("libclang.dll")
            if found_library is not None and not cindex.Config.loaded:
                cindex.Config.set_library_file(found_library)
    elif platform.system() == "Linux":
        # LLVM switched to a monolithical setup that includes everything under
        # /usr/lib/llvm{version_number}/. We glob for the library and select
        # the highest version
        llvm_dir = max(
            (
                path
                for libdir in ["lib64", "lib", "lib32"]
                for path in glob(f"/usr/{libdir}/llvm-*")
                if os.path.exists(str(PurePosixPath(path) / "lib" / "libclang.so.1"))
            ),
            default=None,
            key=_folder_version,
        )

        # Ability to override LLVM/libclang paths
        if "LLVM_DIR_PATH" in os.environ:
            llvm_dir = os.environ["LLVM_DIR_PATH"]

        if "LIBCLANG_PATH" in os.environ:
            libclang_file = os.environ["LIBCLANG_PATH"]
            if not os.path.isfile(libclang_file):
                msg = (
                    f"LIBCLANG_PATH points to {libclang_file!r}, which is not a file. "
                    "Set it to the path of libclang.so.1."
                )
                raise FileNotFoundError(msg)
        elif llvm_dir is not None:
            libclang_file = str(PurePosixPath(llvm_dir) / "lib" / "libclang.so.1")
        else:
            msg = (
                "Failed to find a LLVM installation providing the file "
                "/usr/lib{32,64}/llvm-{VER}/lib/libclang.so.1. Make sure that "
                "you have installed the packages libclang1-{VER} and "
                "libc++-{VER}-dev, where {VER} refers to the desired "
                "Clang/LLVM version (e.g. 11). You may alternatively override "
                "the automatic search by specifying the LLVM_DIR_PATH "
                "(for the LLVM base directory) and/or LIBCLANG_PATH (if "
                "libclang is located at a nonstandard location) environment "
                "variables."
            )
            raise FileNotFoundError(msg)

        if not cindex.Config.loaded:
            cindex.Config.set_library_file(libclang_file)
        cpp_dirs: list[str | None] = []

        if "-stdlib=libc++" not in args:
            cpp_dirs.append(
                max(glob("/usr/include/c++/*"), default=None, key=_folder_version)
            )

            cpp_dirs.append(
                max(
                    glob(f"/usr/include/{platform.machine()}-linux-gnu/c++/*"),
                    default=None,
                    key=_folder_version,
                )
            )
        elif llvm_dir is not None:
            cpp_dirs.append(str(PurePosixPath(llvm_dir) / "include" / "c++" / "v1"))

        if "CLANG_INCLUDE_DIR" in os.environ:
            cpp_dirs.append(os.environ["CLANG_INCLUDE_DIR"])
        elif llvm_dir is not None:
            cpp_dirs.append(
                max(
                    glob(
                        str(PurePosixPath(llvm_dir) / "lib" / "clang" / "*" / "include")
                    ),
                    default=None,
                    key=_folder_version,
                )
            )

        cpp_dirs.append(f"/usr/include/{platform.machine()}-linux-gnu")
        cpp_dirs.append("/usr/include")

        # Capability to specify additional include directories manually
        if "CPP_INCLUDE_DIRS" in os.environ:
            cpp_dirs.extend(
                [
                    cpp_dir
                    for cpp_dir in os.environ["CPP_INCLUDE_DIRS"].split()
                    if os.path.exists(cpp_dir)
                ]
            )

        for cpp_dir in cpp_dirs:
            if cpp_dir is None:
                continue
            parameters.extend(["-isystem", cpp_dir])

    for item in args:
        if item.startswith("-"):
            parameters.append(item)
        else:
            filenames.append(item)

    if len(filenames) == 0:
        msg = "args parameter did not contain any filenames"
        raise NoFilenamesError(msg)

    return parameters, filenames


def extract_all(args: list[str]) -> list[_Doc]:
    parameters, filenames = read_args(args)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = executor.map(_extract_file, filenames, repeat(parameters))
        return [comment for output in results for comment in output]


def write_header(comments: list[_Doc], out_file: TextIO = sys.stdout) -> None:
    print(
        """/*
  This file contains docstrings for use in the Python bindings.
  Do not edit! They were automatically extracted by pybind11_mkdoc.
 */

#define MKD_EXPAND(x)                                      x
#define MKD_COUNT(_1, _2, _3, _4, _5, _6, _7, COUNT, ...)  COUNT
#define MKD_VA_SIZE(...)                                   MKD_EXPAND(MKD_COUNT(__VA_ARGS__, 7, 6, 5, 4, 3, 2, 1, 0))
#define MKD_CAT1(a, b)                                     a ## b
#define MKD_CAT2(a, b)                                     MKD_CAT1(a, b)
#define MKD_DOC1(n1)                                       mkd_doc_##n1
#define MKD_DOC2(n1, n2)                                   mkd_doc_##n1##_##n2
#define MKD_DOC3(n1, n2, n3)                               mkd_doc_##n1##_##n2##_##n3
#define MKD_DOC4(n1, n2, n3, n4)                           mkd_doc_##n1##_##n2##_##n3##_##n4
#define MKD_DOC5(n1, n2, n3, n4, n5)                       mkd_doc_##n1##_##n2##_##n3##_##n4##_##n5
#define MKD_DOC6(n1, n2, n3, n4, n5, n6)                   mkd_doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6
#define MKD_DOC7(n1, n2, n3, n4, n5, n6, n7)               mkd_doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6##_##n7
#define DOC(...)                                           MKD_EXPAND(MKD_EXPAND(MKD_CAT2(MKD_DOC, MKD_VA_SIZE(__VA_ARGS__)))(__VA_ARGS__))

#if defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif
""",
        file=out_file,
    )

    # A suffixed name must not collide with a real symbol of that name, or with an earlier suffixed name.
    taken = {name for name, _, _ in comments}
    name_ctr = 1
    name_prev: str | None = None
    for name, _, comment in sorted(comments, key=lambda x: (x[0], x[1])):
        if name == name_prev:
            name_ctr += 1
            while f"{name_prev}_{name_ctr}" in taken:
                name_ctr += 1
            name = f"{name_prev}_{name_ctr}"
            taken.add(name)
        else:
            name_prev = name
            name_ctr = 1
        sep = "\n" if "\n" in comment else " "
        print(f'\nstatic const char *{name} ={sep}R"doc({comment})doc";', file=out_file)

    print(
        """
#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif
""",
        file=out_file,
    )


def mkdoc(args: list[str], width: int | None, output: str | None = None) -> None:
    if width is not None:
        global docstring_width
        docstring_width = width
    comments = extract_all(args)

    if output:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
            with open(output, "w") as out_file:
                write_header(comments, out_file)
        except:
            # In the event of an error, don't leave a partially-written
            # output file.
            with contextlib.suppress(Exception):
                os.unlink(output)
            raise
    else:
        write_header(comments)
