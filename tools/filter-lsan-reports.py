# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import os
import pathlib
import re
import sys
from typing import List, Optional, Tuple


LEAK_HEADER_RE = re.compile(r"^(Direct|Indirect) leak of .+ allocated from:$")
FRAME_RE = re.compile(r"^\s*#(?P<index>\d+)\s+")
PROJECT_SOURCE_DIRS = (
    "src/sgl/",
    "src/slangpy_ext/",
    "src/slangpy_torch/",
    "tests/",
    "external/slang-rhi/src/",
    "external/slang-rhi/include/",
)
PROJECT_SYMBOL_RE = re.compile(r"\b(?:sgl|rhi|slangpy)::|\bslangpy_ext\b")
PROJECT_BINARY_MARKERS = (
    "sgl_tests+",
    "sgl_tests.exe+",
    "slangpy_ext.pyd+",
    "slangpy_ext.cpython-",
    "libsgl.so+",
    "libsgl.dylib+",
    "sgl.dll+",
)


def normalize_path_text(value: str) -> str:
    return value.replace("\\", "/").lower()


def frame_index(line: str) -> Optional[int]:
    match = FRAME_RE.match(line)
    return int(match.group("index")) if match else None


def is_allocator_interceptor_frame(line: str) -> bool:
    index = frame_index(line)
    if index is None:
        return False

    lower = line.lower()
    allocator_markers = (
        "__interceptor_",
        "asan_malloc",
        "lsan_interceptors",
        "sanitizer_common",
        " in operator new",
        " in operator new[]",
    )
    c_allocator = re.search(r"\bin (?:malloc|calloc|realloc)(?:\s|\(|$)", lower)
    return c_allocator is not None or any(marker in lower for marker in allocator_markers)


def is_project_source_frame(line: str, repo_root: str) -> bool:
    normalized = normalize_path_text(line)
    repo = normalize_path_text(repo_root).rstrip("/") + "/"
    repo_index = normalized.find(repo)
    if repo_index < 0:
        return False

    relative = normalized[repo_index + len(repo) :]
    return relative.startswith(PROJECT_SOURCE_DIRS)


def is_project_binary_frame(line: str) -> bool:
    normalized = normalize_path_text(line)
    return any(marker in normalized for marker in PROJECT_BINARY_MARKERS)


def is_project_leak(block: List[str], repo_root: str) -> bool:
    """Attribute a leak using its first non-allocator frame.

    Looking at every frame makes calls into GPU drivers appear to be project
    leaks merely because SlangPy initiated the external API call. The allocation
    site is the first meaningful frame after allocator interceptors.
    """
    for line in block:
        if frame_index(line) is None:
            continue
        if is_allocator_interceptor_frame(line):
            continue
        return (
            is_project_source_frame(line, repo_root)
            or is_project_binary_frame(line)
            or PROJECT_SYMBOL_RE.search(line) is not None
        )
    return False


def extract_leak_blocks(text: str) -> List[List[str]]:
    lines = text.splitlines()
    blocks: List[List[str]] = []
    index = 0

    while index < len(lines):
        if not LEAK_HEADER_RE.match(lines[index]):
            index += 1
            continue

        block = [lines[index]]
        index += 1
        while index < len(lines):
            line = lines[index]
            if not line.strip():
                break
            if LEAK_HEADER_RE.match(line):
                index -= 1
                break
            if (
                line.startswith("SUMMARY:")
                or line.startswith("----")
                or line.startswith("Suppressions used:")
            ):
                break
            block.append(line)
            index += 1
        blocks.append(block)
        index += 1

    return blocks


def read_log(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify LeakSanitizer reports and surface other sanitizer failures."
    )
    parser.add_argument("--log-dir", type=pathlib.Path)
    parser.add_argument("--repo-root", type=pathlib.Path)
    args = parser.parse_args()

    if args.log_dir:
        log_dir = args.log_dir
    elif os.environ.get("SANITIZER_LOG_DIR"):
        log_dir = pathlib.Path(os.environ["SANITIZER_LOG_DIR"])
    else:
        print("No sanitizer log directory provided and SANITIZER_LOG_DIR is not set.")
        return 1

    repo_root_arg = args.repo_root or pathlib.Path(os.environ.get("GITHUB_WORKSPACE", os.getcwd()))
    repo_root = str(repo_root_arg.resolve())
    if not log_dir.exists():
        print(f"No sanitizer log directory found: {log_dir}")
        return 0

    leak_blocks: List[Tuple[pathlib.Path, List[str]]] = []
    project_root_blocks: List[Tuple[pathlib.Path, List[str]]] = []
    external_root_count = 0
    indirect_block_count = 0
    unexpected_reports: List[Tuple[pathlib.Path, str]] = []

    for log_path in sorted(path for path in log_dir.iterdir() if path.is_file()):
        text = read_log(log_path)
        if not text.strip():
            continue
        if "LeakSanitizer" not in text:
            # LSAN_OPTIONS.log_path is a sanitizer-common setting, so ASan and
            # UBSan diagnostics can land here too. Never silently discard them.
            unexpected_reports.append((log_path, text))
            continue
        blocks = extract_leak_blocks(text)
        if "ERROR: LeakSanitizer:" in text and not blocks:
            # Treat format changes and truncated reports as failures. Silently
            # accepting a report we could not classify would hide regressions.
            unexpected_reports.append((log_path, text))
            continue
        for block in blocks:
            leak_blocks.append((log_path, block))
            if block[0].startswith("Indirect leak"):
                # An indirect allocation is owned by a direct leak root. Its
                # allocation stack alone cannot identify which root retained it.
                indirect_block_count += 1
            elif is_project_leak(block, repo_root):
                project_root_blocks.append((log_path, block))
            else:
                external_root_count += 1

    failed = False

    if unexpected_reports:
        failed = True
        print(f"::error::Found {len(unexpected_reports)} non-LSan sanitizer report(s).")
        for log_path, text in unexpected_reports:
            print(f"\n{log_path}:")
            print(text.rstrip())

    if project_root_blocks:
        failed = True
        print(
            f"::error::LeakSanitizer found {len(project_root_blocks)} direct leak root(s) "
            "attributed to SlangPy or slang-rhi."
        )
        for log_path, block in project_root_blocks:
            print(f"\n{log_path}:")
            print("\n".join(block))

    if external_root_count:
        print(
            f"Ignored {external_root_count} direct LeakSanitizer leak root(s) "
            "attributed to external code."
        )
    if indirect_block_count:
        print(
            f"Ignored {indirect_block_count} indirect LeakSanitizer leak block(s); "
            "their ownership is determined by the direct leak root."
        )
    if not leak_blocks:
        print("No LeakSanitizer leak reports found.")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
