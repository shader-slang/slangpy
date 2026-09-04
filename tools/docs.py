# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Build and validate the SlangPy documentation."""

from __future__ import annotations

import argparse
import importlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


PROJECT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_DIR / "docs"
GENERATED_API_PATH = DOCS_DIR / "generated" / "api.rst"
TUTORIAL_SOURCE_DIR = PROJECT_DIR / "samples" / "tutorials"
TUTORIAL_OUTPUT_DIR = DOCS_DIR / "src" / "tutorials"
DEFAULT_HTML_OUTPUT_DIR = DOCS_DIR / "_build" / "html"

FORBIDDEN_DATA_NAMES = {
    "slangpy.SGL_BUILD_TYPE",
    "slangpy.SGL_GIT_VERSION",
    "slangpy.SHADER_PATH",
    "slangpy.build_dir",
    "slangpy.package_dir",
}
ALLOWED_DATA_NAMES = {
    "slangpy.ALL_LAYERS",
    "slangpy.ALL_MIPS",
    "slangpy.SGL_VERSION_MAJOR",
    "slangpy.SGL_VERSION_MINOR",
    "slangpy.SGL_VERSION_PATCH",
    "slangpy.SGL_VERSION",
}

WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")
POSIX_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![:A-Za-z0-9])/(?:home|Users|tmp|var|opt|workspace|build)(?:/|\b)"
)
BUILD_PATH_RE = re.compile(r"(?:^|[\"'])(?:build|cmake-build-[^/\\]+)[/\\]")
MEMORY_ADDRESS_RE = re.compile(r"\bat 0x[0-9a-fA-F]{6,}\b")


class DocumentationError(RuntimeError):
    """Raised when documentation preparation or validation fails."""


def prepare_tutorials(
    source_dir: Path = TUTORIAL_SOURCE_DIR,
    output_dir: Path = TUTORIAL_OUTPUT_DIR,
) -> None:
    """Copy tutorial sources into the generated Sphinx source directory."""
    if not source_dir.is_dir():
        raise DocumentationError(f"Tutorial source directory does not exist: {source_dir}")

    resolved_output = output_dir.resolve()
    protected_directories = {
        Path(resolved_output.anchor),
        PROJECT_DIR.resolve(),
        DOCS_DIR.resolve(),
        source_dir.resolve(),
    }
    if resolved_output in protected_directories:
        raise DocumentationError(f"Refusing to replace protected directory: {output_dir}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, output_dir)
    print(f"Prepared tutorials in {output_dir}")


def generate_legacy_api() -> None:
    """Generate the legacy API snapshot from a built SlangPy module."""
    sys.path.insert(0, str(PROJECT_DIR))
    sys.path.insert(0, str(DOCS_DIR))
    try:
        importlib.import_module("slangpy")
        generate_api_module = importlib.import_module("generate_api")
        generate_api_module.generate_api()
    except (Exception, SystemExit) as exc:
        raise DocumentationError(
            "Could not import the built SlangPy package to generate API documentation. "
            "Build SlangPy before running documentation validation, or use "
            "'prepare --api-mode snapshot' for an explicit source-only build."
        ) from exc
    finally:
        sys.path.remove(str(DOCS_DIR))
        sys.path.remove(str(PROJECT_DIR))

    print(f"Generated legacy API snapshot at {GENERATED_API_PATH}")


def find_generated_api_problems(text: str) -> list[str]:
    """Return unstable or machine-dependent content found in generated API text."""
    problems: list[str] = []
    for name in sorted(FORBIDDEN_DATA_NAMES):
        if f".. py:data:: {name}" in text:
            problems.append(f"forbidden runtime data entry: {name}")

    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith(".. py:data::"):
            name = stripped.removeprefix(".. py:data::").strip()
            if name not in ALLOWED_DATA_NAMES:
                problems.append(f"line {line_number}: unapproved runtime data entry: {name}")
        if (
            WINDOWS_ABSOLUTE_PATH_RE.search(stripped)
            or POSIX_ABSOLUTE_PATH_RE.search(stripped)
            or BUILD_PATH_RE.search(stripped)
        ):
            problems.append(f"line {line_number}: absolute or build path in generated value")
        if "local changes" in stripped.lower() or "branch:" in stripped.lower():
            problems.append(f"line {line_number}: local Git description in generated value")
        if MEMORY_ADDRESS_RE.search(stripped):
            problems.append(f"line {line_number}: memory address in generated value")
        if stripped.startswith(":value:"):
            value = stripped.removeprefix(":value:").strip()
        else:
            value = ""
        if value.startswith("<") and value.endswith(">"):
            problems.append(f"line {line_number}: unstable object representation in generated value")

    return problems


def check_generated_api(path: Path = GENERATED_API_PATH) -> None:
    """Validate that the generated API snapshot exists and is reproducible."""
    if not path.is_file():
        raise DocumentationError(
            f"Generated API snapshot does not exist: {path}. "
            "Run 'python tools/docs.py prepare --api-mode runtime' after building SlangPy."
        )

    problems = find_generated_api_problems(path.read_text(encoding="utf-8"))
    if problems:
        details = "\n".join(f"  - {problem}" for problem in problems)
        raise DocumentationError(f"Generated API snapshot is not reproducible:\n{details}")

    print(f"Validated generated API snapshot at {path}")


def prepare(api_mode: str) -> None:
    """Prepare tutorials and the selected API input for a Sphinx build."""
    prepare_tutorials()
    if api_mode == "runtime":
        print("API mode: runtime (regenerating from the built SlangPy package)")
        generate_legacy_api()
    elif api_mode == "snapshot":
        print("API mode: snapshot (using the checked-in generated API)")
    else:
        raise DocumentationError(f"Unsupported API mode: {api_mode}")
    check_generated_api()


def run_sphinx(builder: str, output_dir: Path) -> None:
    """Run a strict Sphinx build."""
    environment = os.environ.copy()
    if shutil.which("pandoc", path=environment.get("PATH")) is None:
        try:
            pypandoc = importlib.import_module("pypandoc")
            pandoc_path = Path(pypandoc.get_pandoc_path())
        except (ImportError, OSError) as exc:
            raise DocumentationError(
                "Pandoc is required to render notebooks. Install docs/requirements.txt."
            ) from exc
        environment["PATH"] = f"{pandoc_path.parent}{os.pathsep}{environment.get('PATH', '')}"

    command = [
        sys.executable,
        "-m",
        "sphinx",
        "-W",
        "--keep-going",
        "-b",
        builder,
        str(DOCS_DIR),
        str(output_dir),
    ]
    print(f'Running "{" ".join(command)}" ...')
    result = subprocess.run(command, cwd=PROJECT_DIR, env=environment, check=False)
    if result.returncode != 0:
        raise DocumentationError(f"Sphinx {builder} build failed with exit code {result.returncode}")


def build(api_mode: str, builder: str, output_dir: Path) -> None:
    """Prepare, validate, and build the documentation."""
    prepare(api_mode)
    run_sphinx(builder, output_dir)


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare", help="prepare Sphinx source files")
    prepare_parser.add_argument(
        "--api-mode",
        choices=("runtime", "snapshot"),
        default="runtime",
        help="regenerate API docs from the runtime module or use the checked-in snapshot",
    )

    check_parser = commands.add_parser(
        "check-generated", help="check generated API docs for unstable content"
    )
    check_parser.add_argument(
        "--path", type=Path, default=GENERATED_API_PATH, help="generated API RST file"
    )

    build_parser = commands.add_parser("build", help="prepare and run a strict Sphinx build")
    build_parser.add_argument(
        "--api-mode",
        choices=("runtime", "snapshot"),
        default="runtime",
        help="regenerate API docs from the runtime module or use the checked-in snapshot",
    )
    build_parser.add_argument("--builder", default="html", help="Sphinx builder name")
    build_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_HTML_OUTPUT_DIR,
        help="Sphinx output directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the documentation command-line interface."""
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args.api_mode)
        elif args.command == "check-generated":
            check_generated_api(args.path)
        elif args.command == "build":
            build(args.api_mode, args.builder, args.output_dir)
        else:
            parser.error(f"Unknown command: {args.command}")
    except DocumentationError as exc:
        print(f"Documentation error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
