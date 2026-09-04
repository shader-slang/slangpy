# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest


PROJECT_DIR = Path(__file__).resolve().parents[3]


def load_docs_tool() -> ModuleType:
    """Load tools/docs.py without making tools a Python package."""
    module_name = "slangpy_api_inventory_tool"
    spec = importlib.util.spec_from_file_location(module_name, PROJECT_DIR / "tools" / "docs.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


DOCS_TOOL = load_docs_tool()


def create_sample_package(root: Path) -> tuple[Path, Path]:
    """Create a source/stub package and its reviewed API contract."""
    package_root = root / "sample"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text(
        '''"""Sample package."""

from .model import Other, Widget, Widget as AliasWidget

__all__ = ["AliasWidget", "Other", "Widget"]

raise RuntimeError("The inventory must never import this package")
''',
        encoding="utf-8",
    )
    (package_root / "model.py").write_text(
        '''class Widget:
    """Transform sample values.

    The longer description is preserved separately from the summary.
    """

    def __init__(self, value: object) -> None:
        """Create a widget.

        :param value: Initial value.
        """
        self._value = value

    @property
    def label(self) -> str:
        """A display label."""
        return str(self._value)

    def convert(self, value: object) -> str:
        """Convert a value.

        :param value: Value to convert.
        :return: Text form of the value.
        """
        return str(value)


class Other:
    """An exported but unclassified type."""
''',
        encoding="utf-8",
    )
    (package_root / "model.pyi").write_text(
        '''from typing import overload

class Widget:
    @overload
    def __init__(self, value: int) -> None: ...
    @overload
    def __init__(self, value: str) -> None: ...
    @property
    def label(self) -> str: ...
    @overload
    def convert(self, value: int) -> str: ...
    @overload
    def convert(self, value: str) -> str: ...

class Other: ...
''',
        encoding="utf-8",
    )
    contract = root / "public_api.toml"
    contract.write_text(
        '''schema_version = 1
package = "sample"
package_version = "1.2.3"

[[sections]]
id = "primary"
title = "Primary API"
audience = "user"
names = ["sample.Widget", "sample.AliasWidget"]

[[sections]]
id = "extension"
title = "Extension API"
audience = "extension-author"
names = []
''',
        encoding="utf-8",
    )
    return package_root, contract


def symbol_map(inventory: object) -> dict[str, object]:
    """Index inventory symbols by their published names."""
    return {symbol.name: symbol for symbol in inventory.symbols}


def test_inventory_merges_sources_stubs_overloads_and_aliases(tmp_path: Path) -> None:
    package_root, contract_path = create_sample_package(tmp_path / "first")
    config = DOCS_TOOL.load_public_api(contract_path)

    inventory = DOCS_TOOL.build_inventory(config, package_root)
    symbols = symbol_map(inventory)

    widget = symbols["sample.Widget"]
    alias = symbols["sample.AliasWidget"]
    assert widget.canonical_name == alias.canonical_name == "sample.model.Widget"
    assert widget.aliases == ["sample.AliasWidget"]
    assert alias.aliases == ["sample.Widget"]
    assert len(widget.signatures) == 2
    assert len(symbols["sample.Widget.convert"].signatures) == 2
    assert symbols["sample.Widget.label"].kind == "property"
    assert symbols["sample.Widget.convert"].documentation.parameters == {
        "value": "Value to convert."
    }
    assert symbols["sample.Widget.convert"].documentation.returns == (
        "Text form of the value."
    )
    assert widget.documentation.summary == "Transform sample values."
    assert widget.source.path == "sample/model.py"


def test_inventory_is_byte_stable_across_runs_and_workspaces(tmp_path: Path) -> None:
    first_root, first_contract = create_sample_package(tmp_path / "first")
    second_workspace = tmp_path / "second"
    shutil.copytree(tmp_path / "first", second_workspace)
    second_root = second_workspace / "sample"
    second_contract = second_workspace / "public_api.toml"

    first_inventory = DOCS_TOOL.build_inventory(
        DOCS_TOOL.load_public_api(first_contract), first_root
    )
    second_inventory = DOCS_TOOL.build_inventory(
        DOCS_TOOL.load_public_api(second_contract), second_root
    )
    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"
    DOCS_TOOL.write_inventory(first_inventory, first_output)
    first_bytes = first_output.read_bytes()
    DOCS_TOOL.write_inventory(first_inventory, first_output)
    assert first_output.read_bytes() == first_bytes
    DOCS_TOOL.write_inventory(second_inventory, second_output)
    assert second_output.read_bytes() == first_bytes


def test_report_unclassified_respects_exports_and_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    package_root, contract_path = create_sample_package(tmp_path)

    unclassified = DOCS_TOOL.report_unclassified(contract_path, package_root)

    assert "sample.Other" in unclassified
    assert "sample.Widget" not in unclassified
    assert "sample.AliasWidget" not in unclassified
    assert "sample.Other" in capsys.readouterr().out


def test_inventory_check_detects_stale_output(tmp_path: Path) -> None:
    package_root, contract_path = create_sample_package(tmp_path)
    output = tmp_path / "api.json"
    DOCS_TOOL.inventory_command(contract_path, package_root, output, check=False)
    DOCS_TOOL.inventory_command(contract_path, package_root, output, check=True)
    output.write_text("{}\n", encoding="utf-8")

    with pytest.raises(DOCS_TOOL.DocumentationError, match="missing or stale"):
        DOCS_TOOL.inventory_command(contract_path, package_root, output, check=True)


def test_inventory_rejects_environment_dependent_data(tmp_path: Path) -> None:
    unsafe_values = [
        r"C:\work\slangpy",
        "/home/user/slangpy",
        "build/windows-msvc/Release",
        "<object object at 0x12345678>",
        "branch: local-changes",
    ]
    for unsafe_value in unsafe_values:
        inventory = DOCS_TOOL.ApiInventory(
            schema_version=1,
            package_version="1.2.3",
            sections=[],
            symbols=[
                DOCS_TOOL.ApiSymbol(
                    name="sample.unsafe",
                    canonical_name="sample.unsafe",
                    kind="attribute",
                    section="primary",
                    audience="user",
                    signatures=[unsafe_value],
                )
            ],
        )
        with pytest.raises(DOCS_TOOL.DocumentationError, match="environment-dependent"):
            DOCS_TOOL.write_inventory(inventory, tmp_path / "api.json")


def test_slangpy_pilot_inventory_is_complete_and_environment_independent(
    tmp_path: Path,
) -> None:
    config = DOCS_TOOL.load_public_api(PROJECT_DIR / "docs" / "public_api.toml")
    inventory = DOCS_TOOL.build_inventory(config, PROJECT_DIR / "slangpy")
    symbols = symbol_map(inventory)

    expected_members = {
        "slangpy.Tensor": {"__init__", "device", "from_numpy", "shape"},
        "slangpy.Module": {"__init__", "device", "load_from_file", "link"},
        "slangpy.Function": {"__init__", "call", "map", "return_type"},
        "slangpy.Device": {"__init__", "create_buffer", "info", "wait"},
    }
    for class_name, member_names in expected_members.items():
        assert class_name in symbols
        assert symbols[class_name].kind == "class"
        assert symbols[class_name].signatures
        for member_name in member_names:
            assert f"{class_name}.{member_name}" in symbols

    output = tmp_path / "api.json"
    DOCS_TOOL.write_inventory(inventory, output)
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["package_version"]
    assert not DOCS_TOOL._inventory_problems(output.read_text(encoding="utf-8"))
    assert all(
        source["path"].startswith("slangpy/")
        for symbol in data["symbols"]
        if (source := symbol["source"]) is not None
    )
