# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import copy
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

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


def test_doc_record_parses_pybind11_mkdoc_fields() -> None:
    obj = SimpleNamespace(
        docstring=SimpleNamespace(
            value="""Create a resource.

Parameter ``size``:
    Resource size in bytes.

Returns:
    The new resource.

Raises ``ValueError``:
    If the size is invalid.
"""
        )
    )

    documentation = DOCS_TOOL._doc_record(obj)

    assert documentation.summary == "Create a resource."
    assert documentation.parameters == {"size": "Resource size in bytes."}
    assert documentation.returns == "The new resource."
    assert documentation.raises == ["ValueError: If the size is invalid."]


def test_doc_record_preserves_field_continuations_and_code_indentation() -> None:
    obj = SimpleNamespace(
        docstring=SimpleNamespace(
            value="""Create a resource.

Example:

.. code-block:: python

    resource = create_resource()

:param size: Resource size in
    bytes.
:return: The new
    resource.
"""
        )
    )

    documentation = DOCS_TOOL._doc_record(obj)

    assert documentation.body.endswith("    resource = create_resource()")
    assert documentation.parameters == {"size": "Resource size in bytes."}
    assert documentation.returns == "The new resource."


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
        """from typing import overload

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
""",
        encoding="utf-8",
    )
    contract = root / "public_api.toml"
    contract.write_text(
        """schema_version = 1
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
""",
        encoding="utf-8",
    )
    return package_root, contract


def symbol_map(inventory: Any) -> dict[str, Any]:
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
    assert symbols["sample.Widget.convert"].documentation.returns == ("Text form of the value.")
    assert widget.documentation.summary == "Transform sample values."
    assert widget.source.path == "sample/model.py"


def test_inventory_includes_inherited_members_only_from_unpublished_bases(
    tmp_path: Path,
) -> None:
    package_root, contract_path = create_sample_package(tmp_path)
    model_path = package_root / "model.py"
    model_path.write_text(
        model_path.read_text(encoding="utf-8")
        + "\nclass InternalBase:\n"
        + "    def inherited_api(self) -> None:\n"
        + "        pass\n"
        + "\nclass Child(Widget):\n"
        + "    def child_method(self) -> None:\n"
        + "        pass\n"
        + "\nclass InternalChild(InternalBase):\n"
        + "    pass\n",
        encoding="utf-8",
    )
    init_path = package_root / "__init__.py"
    init_path.write_text(
        init_path.read_text(encoding="utf-8").replace(
            "from .model import Other, Widget, Widget as AliasWidget",
            "from .model import Child, InternalChild, Other, Widget, Widget as AliasWidget",
        ),
        encoding="utf-8",
    )
    contract_path.write_text(
        contract_path.read_text(encoding="utf-8").replace(
            'names = ["sample.Widget", "sample.AliasWidget"]',
            'names = ["sample.Widget", "sample.AliasWidget", "sample.Child", '
            '"sample.InternalChild"]',
        ),
        encoding="utf-8",
    )

    inventory = DOCS_TOOL.build_inventory(DOCS_TOOL.load_public_api(contract_path), package_root)
    symbols = symbol_map(inventory)

    assert "sample.Child.child_method" in symbols
    assert "sample.Child.convert" not in symbols
    assert "sample.InternalChild.inherited_api" in symbols


def test_inventory_normalizes_native_docstring_call_markup(tmp_path: Path) -> None:
    package_root, contract_path = create_sample_package(tmp_path)
    model_path = package_root / "model.py"
    model_path.write_text(
        model_path.read_text(encoding="utf-8").replace(
            '"""Convert a value.', '"""Call ``helper``().'
        ),
        encoding="utf-8",
    )

    inventory = DOCS_TOOL.build_inventory(DOCS_TOOL.load_public_api(contract_path), package_root)

    assert symbol_map(inventory)["sample.Widget.convert"].documentation.summary == (
        "Call ``helper()``."
    )


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


def test_render_sphinx_covers_supported_symbol_shapes(tmp_path: Path) -> None:
    package_root, contract_path = create_sample_package(tmp_path)
    inventory = DOCS_TOOL.build_inventory(DOCS_TOOL.load_public_api(contract_path), package_root)
    inventory.symbols.extend(
        [
            DOCS_TOOL.ApiSymbol(
                name="sample.Mode",
                canonical_name="sample.Mode",
                kind="enum",
                section="primary",
                audience="user",
                documentation=DOCS_TOOL.DocumentationRecord(
                    summary="Select a mode. See :py:class:`sample.Widget`.",
                    status="summary",
                ),
            ),
            DOCS_TOOL.ApiSymbol(
                name="sample.Mode.fast",
                canonical_name="sample.Mode.fast",
                kind="enum-member",
                section="primary",
                audience="user",
                documentation=DOCS_TOOL.DocumentationRecord(
                    summary="Use the fast mode.", status="summary"
                ),
            ),
            DOCS_TOOL.ApiSymbol(
                name="sample.pending",
                canonical_name="sample.pending",
                kind="function",
                section="primary",
                audience="user",
                documentation=DOCS_TOOL.DocumentationRecord(summary="N/A", status="placeholder"),
            ),
        ]
    )
    output_dir = tmp_path / "rendered"

    DOCS_TOOL.render_sphinx(inventory, output_dir, show_missing_documentation=True)

    text = (output_dir / "primary.rst").read_text(encoding="utf-8")
    assert ".. py:class:: sample.Widget" in text
    assert ".. py:property:: sample.Widget.label" in text
    assert ".. py:class:: sample.Mode" in text
    assert ".. py:attribute:: sample.Mode.fast" in text
    assert "**Overloads**" in text
    assert "**Parameters**" in text
    assert "**Returns:**" in text
    assert ":py:obj:`sample.AliasWidget`" in text
    assert ":py:class:`~sample.Widget`" in text
    assert "See :py:class:`sample.Widget`." in text
    assert "*Documentation pending.*" in text
    assert "N/A" not in text


def test_render_sphinx_is_deterministic_and_removes_stale_pages(tmp_path: Path) -> None:
    package_root, contract_path = create_sample_package(tmp_path / "workspace")
    inventory = DOCS_TOOL.build_inventory(DOCS_TOOL.load_public_api(contract_path), package_root)
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    DOCS_TOOL.render_sphinx(inventory, first_output)
    expected = {path.name: path.read_bytes() for path in sorted(first_output.glob("*.rst"))}
    (first_output / "stale.rst").write_text(
        f"{DOCS_TOOL.GENERATED_PAGE_HEADER}\n\nstale\n", encoding="utf-8"
    )
    (first_output / "handwritten.rst").write_text("keep me\n", encoding="utf-8")

    DOCS_TOOL.render_sphinx(inventory, first_output)
    DOCS_TOOL.render_sphinx(inventory, second_output)

    assert not (first_output / "stale.rst").exists()
    assert (first_output / "handwritten.rst").read_text(encoding="utf-8") == "keep me\n"
    assert {
        path.name: path.read_bytes()
        for path in sorted(first_output.glob("*.rst"))
        if path.name != "handwritten.rst"
    } == expected
    assert {path.name: path.read_bytes() for path in sorted(second_output.glob("*.rst"))} == (
        expected
    )
    assert "Documentation pending" not in (first_output / "primary.rst").read_text(encoding="utf-8")


def test_render_sphinx_disambiguates_case_colliding_anchors(tmp_path: Path) -> None:
    inventory = DOCS_TOOL.ApiInventory(
        schema_version=1,
        package_version="1.0.0",
        sections=[{"id": "primary", "title": "Primary", "audience": "user"}],
        symbols=[
            DOCS_TOOL.ApiSymbol(
                name=name,
                canonical_name=name,
                kind="attribute",
                section="primary",
                audience="user",
            )
            for name in ("sample.Record.Field", "sample.Record.field")
        ],
    )

    DOCS_TOOL.render_sphinx(inventory, tmp_path / "rendered")

    anchors = DOCS_TOOL._api_anchors(inventory.symbols)
    assert len(set(anchors.values())) == 2
    assert all(anchor.startswith("api-sample-record-field-") for anchor in anchors.values())
    text = (tmp_path / "rendered" / "primary.rst").read_text(encoding="utf-8")
    assert all(f".. _{anchor}:" in text for anchor in anchors.values())


def test_render_command_validates_contract_and_excludes_unclassified_names(
    tmp_path: Path,
) -> None:
    package_root, contract_path = create_sample_package(tmp_path)
    inventory = DOCS_TOOL.build_inventory(DOCS_TOOL.load_public_api(contract_path), package_root)
    inventory_path = tmp_path / "api.json"
    output_dir = tmp_path / "rendered"
    DOCS_TOOL.write_inventory(inventory, inventory_path)

    DOCS_TOOL.render_command(inventory_path, contract_path, output_dir)

    rendered = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(output_dir.glob("*.rst"))
    )
    assert "sample.Widget" in rendered
    assert "sample.AliasWidget" in rendered
    assert "sample.Other" not in rendered

    inventory.symbols = [
        symbol for symbol in inventory.symbols if symbol.name != "sample.AliasWidget"
    ]
    DOCS_TOOL.write_inventory(inventory, inventory_path)
    with pytest.raises(DOCS_TOOL.DocumentationError, match="missing reviewed public names"):
        DOCS_TOOL.render_command(inventory_path, contract_path, output_dir)


def test_render_command_rejects_broken_internal_reference(tmp_path: Path) -> None:
    package_root, contract_path = create_sample_package(tmp_path)
    inventory = DOCS_TOOL.build_inventory(DOCS_TOOL.load_public_api(contract_path), package_root)
    symbols = symbol_map(inventory)
    symbols["sample.Widget"].documentation = DOCS_TOOL.DocumentationRecord(
        summary="See :py:class:`sample.DoesNotExist`.", status="summary"
    )
    inventory_path = tmp_path / "api.json"
    DOCS_TOOL.write_inventory(inventory, inventory_path)

    with pytest.raises(DOCS_TOOL.DocumentationError, match="broken internal references"):
        DOCS_TOOL.render_command(inventory_path, contract_path, tmp_path / "rendered")


def test_load_inventory_rejects_unknown_schema(tmp_path: Path) -> None:
    package_root, contract_path = create_sample_package(tmp_path)
    inventory = DOCS_TOOL.build_inventory(DOCS_TOOL.load_public_api(contract_path), package_root)
    inventory.schema_version = 999
    inventory_path = tmp_path / "api.json"
    DOCS_TOOL.write_inventory(inventory, inventory_path)

    with pytest.raises(DOCS_TOOL.DocumentationError, match="schema version"):
        DOCS_TOOL.load_inventory(inventory_path)


def create_coverage_inventory() -> Any:
    """Create symbols spanning every mechanical coverage classification."""
    section = "primary"
    audience = "user"

    def symbol(
        name: str,
        documentation: object,
        signatures: list[str] | None = None,
    ) -> object:
        return DOCS_TOOL.ApiSymbol(
            name=f"sample.{name}",
            canonical_name=f"sample.{name}",
            kind="function",
            section=section,
            audience=audience,
            signatures=signatures or [],
            documentation=documentation,
        )

    return DOCS_TOOL.ApiInventory(
        schema_version=1,
        package_version="1.2.3",
        sections=[{"id": section, "title": "Primary", "audience": audience}],
        symbols=[
            symbol(
                "missing",
                DOCS_TOOL.DocumentationRecord(status="missing"),
                ["missing(value: int) -> str"],
            ),
            symbol(
                "placeholder",
                DOCS_TOOL.DocumentationRecord(summary="N/A", status="summary"),
            ),
            symbol(
                "summary",
                DOCS_TOOL.DocumentationRecord(summary="Transform a value.", status="summary"),
                ["summary(value: tuple[int, str], option: str = 'a,b') -> str"],
            ),
            symbol(
                "complete",
                DOCS_TOOL.DocumentationRecord(
                    summary="Transform a value.",
                    parameters={"value": "Input value."},
                    returns="Transformed value.",
                    status="complete",
                ),
                ["complete(value: int) -> str"],
            ),
            symbol(
                "property",
                DOCS_TOOL.DocumentationRecord(summary="Current value.", status="summary"),
            ),
            symbol(
                "example",
                DOCS_TOOL.DocumentationRecord(
                    summary="Demonstrate an operation.",
                    body="Example:\n    run()",
                    status="complete",
                ),
            ),
        ],
    )


def test_calculate_coverage_classifies_requirements_and_examples() -> None:
    report = DOCS_TOOL.calculate_coverage(create_coverage_inventory())
    symbols = {symbol.name: symbol for symbol in report.symbols}

    assert report.counts == {"missing": 1, "placeholder": 1, "summary": 1, "complete": 3}
    assert report.documented == 4
    assert report.score == 7
    assert symbols["sample.summary"].missing_parameters == ["value", "option"]
    assert symbols["sample.summary"].requires_return
    assert not symbols["sample.summary"].return_documented
    assert symbols["sample.complete"].status == "complete"
    assert symbols["sample.property"].status == "complete"
    assert symbols["sample.example"].has_examples


def test_validate_coverage_rejects_each_regression_class(tmp_path: Path) -> None:
    inventory = create_coverage_inventory()
    baseline = DOCS_TOOL.calculate_coverage(inventory)
    baseline_path = tmp_path / "coverage-baseline.json"
    DOCS_TOOL.write_coverage_report(baseline, baseline_path)

    new_missing = copy.deepcopy(inventory)
    new_missing.symbols.append(
        DOCS_TOOL.ApiSymbol(
            name="sample.new_public",
            canonical_name="sample.new_public",
            kind="function",
            section="primary",
            audience="user",
        )
    )
    with pytest.raises(DOCS_TOOL.DocumentationError, match="new public symbols"):
        DOCS_TOOL.validate_coverage(DOCS_TOOL.calculate_coverage(new_missing), baseline_path)

    completed_regression = copy.deepcopy(inventory)
    symbol_map(completed_regression)["sample.complete"].documentation.returns = ""
    with pytest.raises(DOCS_TOOL.DocumentationError, match="completed symbols regressed"):
        DOCS_TOOL.validate_coverage(
            DOCS_TOOL.calculate_coverage(completed_regression), baseline_path
        )

    new_placeholder = copy.deepcopy(inventory)
    symbol_map(new_placeholder)["sample.missing"].documentation = DOCS_TOOL.DocumentationRecord(
        summary="N/A", status="placeholder"
    )
    with pytest.raises(DOCS_TOOL.DocumentationError, match="new placeholders"):
        DOCS_TOOL.validate_coverage(DOCS_TOOL.calculate_coverage(new_placeholder), baseline_path)

    total_regression = copy.deepcopy(inventory)
    symbol_map(total_regression)["sample.summary"].documentation = DOCS_TOOL.DocumentationRecord(
        status="missing"
    )
    with pytest.raises(DOCS_TOOL.DocumentationError, match="documented symbol count decreased"):
        DOCS_TOOL.validate_coverage(DOCS_TOOL.calculate_coverage(total_regression), baseline_path)

    improvement = copy.deepcopy(inventory)
    symbol_map(improvement)["sample.missing"].documentation = DOCS_TOOL.DocumentationRecord(
        summary="New documentation.", status="summary"
    )
    DOCS_TOOL.validate_coverage(DOCS_TOOL.calculate_coverage(improvement), baseline_path)


def test_prepare_agent_site_assets_publishes_versioned_json(tmp_path: Path) -> None:
    package_root, contract_path = create_sample_package(tmp_path / "repository")
    inventory = DOCS_TOOL.build_inventory(DOCS_TOOL.load_public_api(contract_path), package_root)
    inventory_path = tmp_path / "api.json"
    output_dir = tmp_path / "site"
    DOCS_TOOL.write_inventory(inventory, inventory_path)

    DOCS_TOOL.prepare_agent_site_assets(inventory_path, output_dir)

    published_inventory = json.loads((output_dir / "api" / "api.json").read_text(encoding="utf-8"))
    published_coverage = json.loads(
        (output_dir / "api" / "coverage.json").read_text(encoding="utf-8")
    )
    assert published_inventory["schema_version"] == 1
    assert published_coverage["schema_version"] == 1
    assert published_coverage["symbols"]


def test_generate_context_is_bounded_and_repository_relative(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    package_root, contract_path = create_sample_package(repository_root)
    (repository_root / "src" / "sgl").mkdir(parents=True)
    (repository_root / "src" / "sgl" / "widget.h").write_text(
        "class Widget { str convert(int value); };\n", encoding="utf-8"
    )
    (repository_root / "src" / "slangpy_ext").mkdir(parents=True)
    (repository_root / "src" / "slangpy_ext" / "widget.cpp").write_text(
        'type.def("convert", &Widget::convert);\n', encoding="utf-8"
    )
    (repository_root / "slangpy" / "tests").mkdir(parents=True)
    (repository_root / "slangpy" / "tests" / "test_widget.py").write_text(
        "sample.Widget(1).convert(2)\n", encoding="utf-8"
    )
    (repository_root / "samples" / "examples").mkdir(parents=True)
    (repository_root / "samples" / "examples" / "widget.py").write_text(
        "sample.Widget(1).convert(2)\n", encoding="utf-8"
    )
    inventory = DOCS_TOOL.build_inventory(DOCS_TOOL.load_public_api(contract_path), package_root)
    symbol_map(inventory)["sample.Widget.convert"].source = DOCS_TOOL.SourceReference(
        "sample/model.pyi", 8
    )

    text = DOCS_TOOL.generate_context(inventory, "sample.Widget.convert", repository_root)

    assert "# Documentation context: `sample.Widget.convert`" in text
    assert "convert(value: int) -> str" in text
    assert "Value to convert." in text
    assert "## Documentation gaps" in text
    assert "- example" in text
    assert "sample/model.py" in text
    assert "src/sgl/widget.h#L1" in text
    assert "src/slangpy_ext/widget.cpp#L1" in text
    assert "slangpy/tests/test_widget.py#L1" in text
    assert "samples/examples/widget.py#L1" in text
    assert "`sample.Widget`" in text
    assert str(repository_root) not in text


def test_documentation_tasks_follow_api_then_status_priority() -> None:
    config = DOCS_TOOL.PublicApiConfig(
        schema_version=1,
        package="sample",
        package_version="1.0.0",
        sections=(
            DOCS_TOOL.PublicApiSection(
                id="high", title="High", audience="user", names=("sample.High",)
            ),
            DOCS_TOOL.PublicApiSection(
                id="low", title="Low", audience="user", names=("sample.Low",)
            ),
        ),
    )
    inventory = DOCS_TOOL.ApiInventory(
        schema_version=1,
        package_version="1.0.0",
        sections=[
            {"id": "high", "title": "High", "audience": "user"},
            {"id": "low", "title": "Low", "audience": "user"},
        ],
        symbols=[
            DOCS_TOOL.ApiSymbol(
                name="sample.High",
                canonical_name="sample.High",
                kind="class",
                section="high",
                audience="user",
                signatures=["High(value: int) -> None"],
                documentation=DOCS_TOOL.DocumentationRecord(
                    summary="High priority API.", status="summary"
                ),
            ),
            DOCS_TOOL.ApiSymbol(
                name="sample.High.missing",
                canonical_name="sample.High.missing",
                kind="method",
                section="high",
                audience="user",
            ),
            DOCS_TOOL.ApiSymbol(
                name="sample.Low",
                canonical_name="sample.Low",
                kind="class",
                section="low",
                audience="user",
            ),
        ],
    )

    tasks = DOCS_TOOL.calculate_documentation_tasks(inventory, config)

    assert [task.name for task in tasks] == [
        "sample.High.missing",
        "sample.High",
        "sample.Low",
    ]
    assert tasks[0].missing_fields == ["summary", "example"]
    assert tasks[1].missing_fields == ["parameters: value", "example"]
    assert [
        task.name for task in DOCS_TOOL.calculate_documentation_tasks(inventory, config, "low")
    ] == ["sample.Low"]
    json_output = json.loads(DOCS_TOOL._tasks_json(inventory, tasks, None))
    assert json_output["schema_version"] == 1
    assert json_output["count"] == 3
    markdown = DOCS_TOOL._tasks_markdown(inventory, tasks, None)
    assert "`sample.High.missing`" in markdown
    with pytest.raises(DOCS_TOOL.DocumentationError, match="Unknown public API section"):
        DOCS_TOOL.calculate_documentation_tasks(inventory, config, "unknown")


def test_validate_agent_outputs_covers_representative_page_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs_dir = tmp_path / "docs"
    output_dir = tmp_path / "html"
    source_pages = {
        "src/basics/firstfunctions.rst": "Your First Function\n===================\n",
        "generated/api/device.rst": "Device\n======\n",
        "generated/api/functional-api.rst": "Functional API\n==============\n",
        "src/tutorials/compute_shader.ipynb": "{}\n",
    }
    markdown_pages = {
        "src/basics/firstfunctions.md": "# Your First Function\n\n```python\npass\n```\n",
        "generated/api/device.md": (
            "# Device\n\n`slangpy.Device`\n\n```python\nDevice()\n```\n\n"
            "**Source:** [device.pyi](slangpy/device.pyi#L1)\n"
        ),
        "generated/api/functional-api.md": (
            "# Functional API\n\n[`slangpy.Module`](#slangpy.Module)\n\n"
            "```python\nModule()\n```\n"
        ),
        "src/tutorials/compute_shader.md": "# Compute Shader\n\n```c#\n[numthreads(1, 1, 1)]\n```\n",
    }
    for relative, content in source_pages.items():
        path = docs_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    for relative, content in markdown_pages.items():
        path = output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    inventory = DOCS_TOOL.ApiInventory(
        schema_version=1,
        package_version="1.0.0",
        sections=[],
        symbols=[],
    )
    DOCS_TOOL.write_inventory(inventory, output_dir / "api" / "api.json")
    DOCS_TOOL.write_coverage_report(
        DOCS_TOOL.calculate_coverage(inventory), output_dir / "api" / "coverage.json"
    )
    monkeypatch.setattr(DOCS_TOOL, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(DOCS_TOOL, "GENERATED_API_PATH", docs_dir / "generated" / "api.rst")
    DOCS_TOOL.write_curated_llms(output_dir)

    DOCS_TOOL.validate_agent_outputs(output_dir)

    (output_dir / "src" / "tutorials" / "compute_shader.md").unlink()
    with pytest.raises(DOCS_TOOL.DocumentationError, match="missing pages"):
        DOCS_TOOL.validate_agent_outputs(output_dir)


def test_slangpy_reviewed_inventory_is_complete_and_environment_independent(
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

    expected_sections = {
        "application",
        "constants",
        "core",
        "device",
        "extension-api",
        "functional-api",
        "logging",
        "math",
        "platform",
        "slangpy",
        "threading",
        "ui",
        "utilities",
        "windowing",
    }
    assert {section.id for section in config.sections} == expected_sections
    legacy_order = json.loads((PROJECT_DIR / "docs" / "api_order.json").read_text(encoding="utf-8"))
    discovered = DOCS_TOOL._StaticPackage(
        PROJECT_DIR / "slangpy", config.package
    ).discover_public_names()
    legacy_matches = {
        name
        for patterns in legacy_order.values()
        for pattern in patterns
        for name in discovered
        if re.fullmatch(pattern, name)
    }
    assert legacy_matches <= set(symbols)
    assert "slangpy.Buffer.copy_from_numpy" in symbols
    assert "slangpy.CommandEncoder.begin_compute_pass" in symbols
    assert "slangpy.float4.x" in symbols
    assert "slangpy.ui.Window.title" in symbols
    assert "slangpy.DiffPair" in symbols
    assert "slangpy.build_dir" not in symbols

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


def test_phase5_primary_workflows_are_documented_and_have_tested_examples() -> None:
    config = DOCS_TOOL.load_public_api(PROJECT_DIR / "docs" / "public_api.toml")
    inventory = DOCS_TOOL.build_inventory(config, PROJECT_DIR / "slangpy")
    coverage = {symbol.name: symbol for symbol in DOCS_TOOL.calculate_coverage(inventory).symbols}

    primary_classes = (
        "slangpy.Tensor",
        "slangpy.Module",
        "slangpy.Function",
        "slangpy.DiffPair",
    )
    for name in primary_classes:
        assert coverage[name].status == "complete"
        assert coverage[name].has_examples
        assert all(
            symbol.status != "placeholder"
            for symbol_name, symbol in coverage.items()
            if symbol_name == name or symbol_name.startswith(f"{name}.")
        )

    campaign_entry_points = (
        "slangpy.create_device",
        "slangpy.create_torch_device",
        "slangpy.diffPair",
        "slangpy.floatDiffPair",
        "slangpy.Buffer",
        "slangpy.Buffer.copy_from_numpy",
        "slangpy.Buffer.to_numpy",
        "slangpy.Texture",
        "slangpy.Texture.copy_from_numpy",
        "slangpy.Texture.to_numpy",
        "slangpy.Device.create_buffer",
        "slangpy.Device.create_texture",
        "slangpy.Device.load_module",
        "slangpy.Device.load_module_from_source",
        "slangpy.Device.load_program",
        "slangpy.SlangSession.load_module",
        "slangpy.SlangSession.load_module_from_source",
        "slangpy.SlangSession.load_program",
        "slangpy.SlangModule",
    )
    assert all(coverage[name].status == "complete" for name in campaign_entry_points)

    example_tests = (PROJECT_DIR / "samples" / "tests" / "examples" / "test_examples.py").read_text(
        encoding="utf-8"
    )
    for test_name in ("test_buffers", "test_first_function_scalar", "test_textures"):
        assert f"def {test_name}(" in example_tests
