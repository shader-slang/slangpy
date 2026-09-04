# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Build and validate the SlangPy documentation."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import griffe

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.9/3.10
    import tomli as tomllib  # pyright: ignore[reportMissingImports]


PROJECT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_DIR / "docs"
GENERATED_API_PATH = DOCS_DIR / "generated" / "api.rst"
TUTORIAL_SOURCE_DIR = PROJECT_DIR / "samples" / "tutorials"
TUTORIAL_OUTPUT_DIR = DOCS_DIR / "src" / "tutorials"
DEFAULT_HTML_OUTPUT_DIR = DOCS_DIR / "_build" / "html"
PUBLIC_API_PATH = DOCS_DIR / "public_api.toml"
API_INVENTORY_PATH = DOCS_DIR / "api" / "api.json"

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
OBJECT_REPRESENTATION_RE = re.compile(
    r"<(?:[^>]*\bobject\b[^>]*|[^>]*\b(?:built-in )?function\b[^>]*)>"
)


class DocumentationError(RuntimeError):
    """Raised when documentation preparation or validation fails."""


@dataclass(frozen=True)
class PublicApiSection:
    """One ordered section in the reviewed public API contract."""

    id: str
    title: str
    audience: str
    names: tuple[str, ...] = ()
    patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class PublicApiConfig:
    """Reviewed public API configuration loaded from TOML."""

    schema_version: int
    package: str
    sections: tuple[PublicApiSection, ...]
    version_source: str | None = None
    package_version: str | None = None


@dataclass
class DocumentationRecord:
    """Normalized documentation attached to an API symbol."""

    summary: str = ""
    body: str = ""
    parameters: dict[str, str] = field(default_factory=dict)
    returns: str = ""
    raises: list[str] = field(default_factory=list)
    status: str = "missing"


@dataclass
class SourceReference:
    """Repository-relative source location for an API symbol."""

    path: str
    line: int | None = None


@dataclass
class ApiSymbol:
    """Stable, renderer-independent representation of one public symbol."""

    name: str
    canonical_name: str
    kind: str
    section: str
    audience: str
    signatures: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    documentation: DocumentationRecord = field(default_factory=DocumentationRecord)
    source: SourceReference | None = None


@dataclass
class ApiInventory:
    """Versioned structured public API snapshot."""

    schema_version: int
    package_version: str
    sections: list[dict[str, str]]
    symbols: list[ApiSymbol]


@dataclass(frozen=True)
class _ObjectPair:
    """Source and stub views of the same published object."""

    published_name: str
    canonical_name: str
    source: Any | None
    stub: Any | None


class _StaticPackage:
    """Load Python sources and generated stubs without importing the package."""

    def __init__(self, package_root: Path, package_name: str) -> None:
        super().__init__()
        self.package_root = package_root.resolve()
        self.package_name = package_name
        try:
            self.source = griffe.load(
                package_name,
                search_paths=[self.package_root.parent],
                allow_inspection=False,
                resolve_aliases=True,
                docstring_parser="sphinx",
            )
        except Exception as exc:
            raise DocumentationError(
                f"Could not statically load package '{package_name}' from {package_root}"
            ) from exc

        self.stub_modules: dict[str, Any] = {}
        for path in sorted(self.package_root.rglob("*.pyi")):
            relative = path.relative_to(self.package_root)
            if path.name == "__init__.pyi":
                suffix = relative.parent.parts
            else:
                suffix = (*relative.parent.parts, path.stem)
            module_name = ".".join((package_name, *suffix))
            self.stub_modules[module_name] = griffe.visit(
                module_name,
                path,
                path.read_text(encoding="utf-8"),
                docstring_parser="sphinx",
            )

    @staticmethod
    def _final_target(obj: Any | None) -> Any | None:
        if obj is None:
            return None
        if getattr(obj, "is_alias", False):
            try:
                return obj.final_target
            except Exception:
                return None
        return obj

    def source_object(self, name: str) -> Any | None:
        """Return a source object for a fully qualified published name."""
        parts = name.split(".")
        if not parts or parts[0] != self.package_name:
            return None
        current: Any = self.source
        for part in parts[1:]:
            current = self._final_target(current)
            if current is None:
                return None
            current = getattr(current, "members", {}).get(part)
            if current is None:
                return None
        return self._final_target(current)

    def stub_object(self, name: str, seen: set[str] | None = None) -> Any | None:
        """Return a stub object, resolving aliases between generated stub modules."""
        if seen is None:
            seen = set()
        if name in seen:
            return None
        seen.add(name)

        parts = name.split(".")
        for index in range(len(parts), 0, -1):
            module_name = ".".join(parts[:index])
            module = self.stub_modules.get(module_name)
            if module is None:
                continue
            current: Any = module
            for part in parts[index:]:
                current = getattr(current, "members", {}).get(part)
                if current is None:
                    break
            if current is None:
                continue
            if getattr(current, "is_alias", False):
                return self.stub_object(str(current.target_path), seen)
            return current
        return None

    def object_pair(self, published_name: str) -> _ObjectPair | None:
        """Return the combined source/stub representation for a published name."""
        source = self.source_object(published_name)
        canonical_name = str(getattr(source, "canonical_path", published_name))
        stub = self.stub_object(published_name)
        if stub is None:
            stub = self.stub_object(canonical_name)
        if source is None and stub is None:
            return None
        if source is None:
            canonical_name = str(getattr(stub, "canonical_path", published_name))
        return _ObjectPair(published_name, canonical_name, source, stub)

    def discover_public_names(self) -> list[str]:
        """Return statically reachable public-looking package-level names."""
        discovered: set[str] = set()
        visited_modules: set[str] = set()

        def visit_module(module: Any, published_path: str) -> None:
            target = self._final_target(module)
            if target is None:
                return
            canonical_path = str(getattr(target, "canonical_path", published_path))
            if canonical_path in visited_modules:
                return
            visited_modules.add(canonical_path)
            exports = getattr(target, "exports", None)
            explicit_exports = {str(export) for export in exports} if exports is not None else None
            for member_name, member in getattr(target, "members", {}).items():
                if member_name.startswith("_"):
                    continue
                if explicit_exports is not None and member_name not in explicit_exports:
                    continue
                member_path = f"{published_path}.{member_name}"
                member_target = self._final_target(member)
                if member_target is None:
                    continue
                if getattr(member_target, "is_module", False) and str(
                    getattr(member_target, "canonical_path", "")
                ).startswith(f"{self.package_name}."):
                    visit_module(member_target, member_path)
                else:
                    discovered.add(member_path)

        visit_module(self.source, self.package_name)
        return sorted(discovered)


def load_public_api(path: Path) -> PublicApiConfig:
    """Load and validate a public API contract from TOML."""
    if not path.is_file():
        raise DocumentationError(f"Public API contract does not exist: {path}")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        sections = tuple(
            PublicApiSection(
                id=str(section["id"]),
                title=str(section["title"]),
                audience=str(section["audience"]),
                names=tuple(str(name) for name in section.get("names", [])),
                patterns=tuple(str(pattern) for pattern in section.get("patterns", [])),
            )
            for section in data["sections"]
        )
        config = PublicApiConfig(
            schema_version=int(data["schema_version"]),
            package=str(data["package"]),
            version_source=data.get("version_source"),
            package_version=data.get("package_version"),
            sections=sections,
        )
    except (KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise DocumentationError(f"Invalid public API contract: {path}") from exc

    if config.schema_version != 1:
        raise DocumentationError(f"Unsupported public API schema version: {config.schema_version}")
    if not config.sections or len({section.id for section in config.sections}) != len(
        config.sections
    ):
        raise DocumentationError("Public API sections must be non-empty and have unique IDs")
    if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", config.package):
        raise DocumentationError(f"Invalid public API package name: {config.package}")
    reviewed_names: dict[str, str] = {}
    for section in config.sections:
        if not section.id or not section.title:
            raise DocumentationError("Public API section IDs and titles must be non-empty")
        if section.audience not in {"user", "extension-author"}:
            raise DocumentationError(
                f"Unsupported audience in section '{section.id}': {section.audience}"
            )
        for name in section.names:
            if not name.startswith(f"{config.package}."):
                raise DocumentationError(
                    f"Public API name '{name}' is outside package '{config.package}'"
                )
            if name in reviewed_names:
                raise DocumentationError(
                    f"Public API name '{name}' occurs in both '{reviewed_names[name]}' "
                    f"and '{section.id}'"
                )
            reviewed_names[name] = section.id
        for pattern in section.patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise DocumentationError(
                    f"Invalid public API pattern in section '{section.id}': {pattern}"
                ) from exc
    return config


def _package_version(config: PublicApiConfig, package_root: Path) -> str:
    if config.package_version is not None:
        return config.package_version
    if config.version_source is None:
        return "unknown"
    version_path = package_root.parent / config.version_source
    if not version_path.is_file():
        raise DocumentationError(f"Package version source does not exist: {version_path}")
    text = version_path.read_text(encoding="utf-8")
    components = []
    for component in ("MAJOR", "MINOR", "PATCH"):
        match = re.search(rf"^#define SGL_VERSION_{component} (\d+)$", text, re.MULTILINE)
        if match is None:
            raise DocumentationError(f"Could not read SGL_VERSION_{component} from {version_path}")
        components.append(match.group(1))
    return ".".join(components)


def _doc_record(*objects: Any | None) -> DocumentationRecord:
    raw = ""
    for obj in objects:
        docstring = getattr(obj, "docstring", None)
        value = getattr(docstring, "value", "") if docstring is not None else ""
        if value and str(value).strip():
            raw = str(value).replace("\r\n", "\n").strip()
            break
    if not raw:
        return DocumentationRecord()
    if raw.strip().upper() == "N/A":
        return DocumentationRecord(status="placeholder")

    parameters: dict[str, str] = {}
    raises: list[str] = []
    returns = ""
    narrative: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        parameter = re.match(r":param\s+([^:]+):\s*(.*)", stripped)
        returned = re.match(r":returns?:\s*(.*)", stripped)
        raised = re.match(r":raises?\s+([^:]+):\s*(.*)", stripped)
        if parameter:
            parameters[parameter.group(1)] = parameter.group(2).strip()
        elif returned:
            returns = returned.group(1).strip()
        elif raised:
            raises.append(f"{raised.group(1)}: {raised.group(2).strip()}".rstrip(": "))
        elif not stripped.startswith(":type") and not stripped.startswith(":rtype"):
            narrative.append(line.rstrip())

    narrative_text = "\n".join(narrative).strip()
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", narrative_text) if part.strip()]
    summary = " ".join(paragraphs[0].splitlines()) if paragraphs else ""
    body = "\n\n".join(paragraphs[1:])
    status = "complete" if body or parameters or returns or raises else "summary"
    return DocumentationRecord(summary, body, parameters, returns, raises, status)


def _overloads(parent: Any | None, name: str) -> list[Any]:
    if parent is None:
        return []
    overloads = getattr(parent, "overloads", {})
    return list(overloads.get(name, []))


def _member(parent: Any | None, name: str) -> Any | None:
    if parent is None:
        return None
    try:
        members = parent.all_members
    except (AttributeError, KeyError):
        members = getattr(parent, "members", {})
    obj = members.get(name)
    return _StaticPackage._final_target(obj)


def _signatures(obj: Any | None, overloads: list[Any] | None = None) -> list[str]:
    functions = overloads or []
    if not functions and obj is not None:
        functions = list(getattr(obj, "overloads", []) or []) or [obj]
    result: list[str] = []
    for function in functions:
        signature = getattr(function, "signature", None)
        if callable(signature):
            result.append(str(signature()))
    return list(dict.fromkeys(result))


def _kind(obj: Any | None, name: str) -> str:
    if name == "__init__":
        return "constructor"
    if obj is None:
        return "method"
    if getattr(obj, "is_class", False):
        return "class"
    labels = set(getattr(obj, "labels", set()))
    if "property" in labels:
        return "property"
    if "staticmethod" in labels:
        return "staticmethod"
    if "classmethod" in labels:
        return "classmethod"
    if getattr(obj, "is_function", False):
        parent = getattr(obj, "parent", None)
        return "function" if getattr(parent, "is_module", False) else "method"
    return "attribute"


def _source_reference(obj: Any | None, repository_root: Path) -> SourceReference | None:
    if obj is None:
        return None
    filepath = getattr(obj, "filepath", None)
    if not isinstance(filepath, Path) or not filepath.is_file():
        return None
    line = getattr(obj, "lineno", None)
    if filepath.suffix == ".py" and isinstance(line, int):
        line_count = len(filepath.read_text(encoding="utf-8").splitlines())
        if line > line_count:
            return None
    try:
        relative = filepath.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return None
    return SourceReference(relative, line if isinstance(line, int) else None)


def _class_symbols(
    pair: _ObjectPair,
    section: PublicApiSection,
    repository_root: Path,
) -> list[ApiSymbol]:
    source = pair.source
    stub = pair.stub
    documentation = _doc_record(source, stub)
    source_reference = _source_reference(source, repository_root) or _source_reference(
        stub, repository_root
    )
    constructor_overloads = _overloads(stub, "__init__") or _overloads(source, "__init__")
    constructor = _member(stub, "__init__")
    if constructor is None:
        constructor = _member(source, "__init__")
    class_signatures = [
        signature.replace("__init__", pair.published_name.rsplit(".", 1)[-1], 1)
        for signature in _signatures(constructor, constructor_overloads)
    ]
    symbols = [
        ApiSymbol(
            name=pair.published_name,
            canonical_name=pair.canonical_name,
            kind="class",
            section=section.id,
            audience=section.audience,
            signatures=class_signatures,
            documentation=documentation,
            source=source_reference,
        )
    ]

    member_names: set[str] = set()
    for parent in (source, stub):
        if parent is None:
            continue
        try:
            member_names.update(parent.all_members)
        except (AttributeError, KeyError):
            member_names.update(getattr(parent, "members", {}))
        member_names.update(getattr(parent, "overloads", {}))

    allowed_special = {
        "__call__",
        "__enter__",
        "__exit__",
        "__getitem__",
        "__init__",
        "__iter__",
        "__len__",
        "__repr__",
        "__setitem__",
        "__str__",
    }
    ordered_names = sorted(
        (name for name in member_names if not name.startswith("_") or name in allowed_special),
        key=lambda name: (name != "__init__", name),
    )
    for name in ordered_names:
        source_member = _member(source, name)
        stub_member = _member(stub, name)
        overloads = _overloads(stub, name) or _overloads(source, name)
        representative = source_member
        if representative is None:
            representative = stub_member
        if representative is None and overloads:
            representative = overloads[0]
        if representative is None:
            continue
        canonical_name = str(
            getattr(
                source_member if source_member is not None else stub_member,
                "canonical_path",
                f"{pair.canonical_name}.{name}",
            )
        )
        symbols.append(
            ApiSymbol(
                name=f"{pair.published_name}.{name}",
                canonical_name=canonical_name,
                kind=_kind(representative, name),
                section=section.id,
                audience=section.audience,
                signatures=_signatures(
                    stub_member if stub_member is not None else source_member,
                    overloads,
                ),
                documentation=_doc_record(source_member, stub_member, *overloads),
                source=_source_reference(source_member, repository_root)
                or _source_reference(stub_member, repository_root)
                or _source_reference(representative, repository_root),
            )
        )
    return symbols


def build_inventory(config: PublicApiConfig, package_root: Path) -> ApiInventory:
    """Build a structured inventory from static Python sources and generated stubs."""
    package = _StaticPackage(package_root, config.package)
    discovered = package.discover_public_names()
    selected: set[str] = set()
    symbols: list[ApiSymbol] = []

    for section in config.sections:
        section_names = list(section.names)
        for pattern in section.patterns:
            section_names.extend(name for name in discovered if re.fullmatch(pattern, name))
        for name in dict.fromkeys(section_names):
            if name in selected:
                continue
            pair = package.object_pair(name)
            if pair is None:
                raise DocumentationError(
                    f"Public API symbol '{name}' from section '{section.id}' was not found"
                )
            selected.add(name)
            representative = pair.source if pair.source is not None else pair.stub
            if getattr(representative, "is_class", False):
                symbols.extend(_class_symbols(pair, section, package_root.parent))
            else:
                symbols.append(
                    ApiSymbol(
                        name=name,
                        canonical_name=pair.canonical_name,
                        kind=_kind(representative, name.rsplit(".", 1)[-1]),
                        section=section.id,
                        audience=section.audience,
                        signatures=_signatures(pair.stub if pair.stub is not None else pair.source),
                        documentation=_doc_record(pair.source, pair.stub),
                        source=_source_reference(pair.source, package_root.parent)
                        or _source_reference(pair.stub, package_root.parent),
                    )
                )

    inventory = ApiInventory(
        schema_version=1,
        package_version=_package_version(config, package_root),
        sections=[
            {"id": section.id, "title": section.title, "audience": section.audience}
            for section in config.sections
        ],
        symbols=symbols,
    )
    return normalize_inventory(inventory, package_root.parent)


def normalize_inventory(inventory: ApiInventory, repository_root: Path) -> ApiInventory:
    """Normalize ordering, aliases, paths, and repeated values in an inventory."""
    del repository_root  # Paths are made relative while symbols are extracted.
    aliases_by_canonical: dict[str, set[str]] = {}
    for symbol in inventory.symbols:
        aliases_by_canonical.setdefault(symbol.canonical_name, set()).add(symbol.name)
    for symbol in inventory.symbols:
        symbol.signatures = list(dict.fromkeys(symbol.signatures))
        symbol.aliases = sorted(aliases_by_canonical[symbol.canonical_name] - {symbol.name})
        if symbol.source is not None:
            symbol.source.path = symbol.source.path.replace("\\", "/")
    return inventory


def _inventory_text(inventory: ApiInventory) -> str:
    return json.dumps(asdict(inventory), indent=2, ensure_ascii=True) + "\n"


def _inventory_problems(text: str) -> list[str]:
    problems: list[str] = []
    checks = {
        "Windows absolute path": WINDOWS_ABSOLUTE_PATH_RE,
        "POSIX absolute path": POSIX_ABSOLUTE_PATH_RE,
        "build path": BUILD_PATH_RE,
        "memory address": MEMORY_ADDRESS_RE,
        "object representation": OBJECT_REPRESENTATION_RE,
    }
    for label, pattern in checks.items():
        if pattern.search(text):
            problems.append(label)
    if "local changes" in text.lower() or "branch:" in text.lower():
        problems.append("local Git description")
    return problems


def write_inventory(inventory: ApiInventory, output: Path) -> None:
    """Write an inventory only when its normalized JSON content changes."""
    text = _inventory_text(inventory)
    problems = _inventory_problems(text)
    if problems:
        raise DocumentationError(
            "API inventory contains environment-dependent data: " + ", ".join(problems)
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists() or output.read_text(encoding="utf-8") != text:
        output.write_text(text, encoding="utf-8")


def inventory_command(config_path: Path, package_root: Path, output: Path, check: bool) -> None:
    """Generate or check the structured public API snapshot."""
    config = load_public_api(config_path)
    inventory = build_inventory(config, package_root)
    text = _inventory_text(inventory)
    problems = _inventory_problems(text)
    if problems:
        raise DocumentationError(
            "API inventory contains environment-dependent data: " + ", ".join(problems)
        )
    if check:
        if not output.is_file() or output.read_text(encoding="utf-8") != text:
            raise DocumentationError(
                f"API inventory is missing or stale: {output}. Run 'python tools/docs.py inventory'."
            )
        action = "Validated"
    else:
        write_inventory(inventory, output)
        action = "Wrote"
    placeholder_count = sum(
        symbol.documentation.status in {"missing", "placeholder"} for symbol in inventory.symbols
    )
    print(
        f"{action} API inventory at {output}: {len(inventory.symbols)} symbols, "
        f"{placeholder_count} missing or placeholder documentation records"
    )


def report_unclassified(config_path: Path, package_root: Path) -> list[str]:
    """Print and return reachable public-looking names outside the reviewed contract."""
    config = load_public_api(config_path)
    package = _StaticPackage(package_root, config.package)
    discovered = package.discover_public_names()

    def classified(name: str) -> bool:
        return any(
            name in section.names
            or any(re.fullmatch(pattern, name) for pattern in section.patterns)
            for section in config.sections
        )

    unclassified = [name for name in discovered if not classified(name)]
    print(f"Unclassified public-looking names: {len(unclassified)}")
    for name in unclassified:
        print(name)
    return unclassified


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
            problems.append(
                f"line {line_number}: unstable object representation in generated value"
            )

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
        raise DocumentationError(
            f"Sphinx {builder} build failed with exit code {result.returncode}"
        )


def build(api_mode: str, builder: str, output_dir: Path) -> None:
    """Prepare, validate, and build the documentation."""
    prepare(api_mode)
    run_sphinx(builder, output_dir)


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    inventory_parser = commands.add_parser(
        "inventory", help="generate or validate the structured public API inventory"
    )
    inventory_parser.add_argument(
        "--config", type=Path, default=PUBLIC_API_PATH, help="public API TOML contract"
    )
    inventory_parser.add_argument(
        "--package-root", type=Path, default=PROJECT_DIR / "slangpy", help="package directory"
    )
    inventory_parser.add_argument(
        "--output", type=Path, default=API_INVENTORY_PATH, help="output JSON inventory"
    )
    inventory_parser.add_argument(
        "--check", action="store_true", help="fail if the checked-in inventory is stale"
    )

    unclassified_parser = commands.add_parser(
        "report-unclassified", help="report public-looking names outside the API contract"
    )
    unclassified_parser.add_argument(
        "--config", type=Path, default=PUBLIC_API_PATH, help="public API TOML contract"
    )
    unclassified_parser.add_argument(
        "--package-root", type=Path, default=PROJECT_DIR / "slangpy", help="package directory"
    )

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
        if args.command == "inventory":
            inventory_command(args.config, args.package_root, args.output, args.check)
        elif args.command == "report-unclassified":
            report_unclassified(args.config, args.package_root)
        elif args.command == "prepare":
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
