# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest
import slangpy as spy


@pytest.fixture
def generator() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "docs" / "generate_api.py"
    spec = importlib.util.spec_from_file_location("slangpy_docs_generator_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_custom_module_filters_and_output(
    generator: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = ModuleType("example_api")
    child = ModuleType("example_api.child")
    excluded = ModuleType("example_api.hidden")
    unrelated = ModuleType("example_api_other")
    root.KEEP = 1
    root.DROP = 2
    root.child = child
    root.hidden = excluded
    root.unrelated = unrelated
    child.VALUE = 3
    child.parent = root
    excluded.SECRET = 4
    unrelated.FOREIGN = 5
    monkeypatch.setitem(sys.modules, root.__name__, root)
    monkeypatch.setattr(generator, "DIR", tmp_path / "missing-defaults")
    target = tmp_path / "output" / "reference.rst"

    generator.generate_api(
        root.__name__,
        api_order={"Child": [r"example_api.child\..*"], "Root": [r"example_api.KEEP"]},
        output_path=target,
        include_module=lambda name: name != excluded.__name__,
        include_member=lambda name, obj: not name.endswith(".DROP"),
    )

    text = target.read_text(encoding="utf-8")
    assert "example_api.KEEP" in text
    assert "example_api.child.VALUE" in text
    assert text.index("Child\n") < text.index("Root\n")
    assert "DROP" not in text and "SECRET" not in text and "FOREIGN" not in text
    assert not (generator.DIR / "generated" / "api.rst").exists()

    generator.generate_api(root.__name__, api_order={}, output_path=target)
    text = target.read_text(encoding="utf-8")
    assert "example_api.DROP" in text and "example_api.hidden.SECRET" in text
    assert "FOREIGN" not in text
    del root.DROP
    generator.generate_api(root.__name__, api_order={}, output_path=target)
    assert "example_api.DROP" not in target.read_text(encoding="utf-8")
    modified = target.stat().st_mtime_ns
    generator.generate_api(root.__name__, api_order={}, output_path=target)
    assert target.stat().st_mtime_ns == modified


def test_default_order_and_output(
    generator: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = ModuleType("slangpy")
    root.VALUE = 42
    monkeypatch.setattr(generator, "DIR", tmp_path)
    monkeypatch.setattr(generator.importlib, "import_module", lambda name: root)
    (tmp_path / "api_order.json").write_text(json.dumps({"Values": [r"slangpy.VALUE"]}))
    generator.generate_api()
    text = (tmp_path / "generated/api.rst").read_text(encoding="utf-8")
    assert text.startswith("Values\n")
    assert ".. py:data:: slangpy.VALUE" in text


def test_native_renderer_retains_overloads_properties_and_enums(generator: ModuleType) -> None:
    ctx = generator.Context()
    ctx.push("slangpy", indent=False)
    generator.process_class(spy.Bitmap, "Bitmap", ctx)
    generator.process_class(spy.DeviceType, "DeviceType", ctx)
    assert ctx.output.count(".. py:method:: __init__(") > 1
    assert ".. py:property:: width" in ctx.output
    assert ".. py:attribute:: slangpy.DeviceType.d3d12" in ctx.output
    assert ":no-index:" in ctx.output


def test_import_failure_leaves_existing_output(
    generator: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "reference.rst"
    target.write_text("existing reference", encoding="utf-8")

    def fail_import(name: str) -> ModuleType:
        raise ImportError("missing native bindings")

    monkeypatch.setattr(generator.importlib, "import_module", fail_import)
    with pytest.raises(ImportError, match="missing native bindings"):
        generator.generate_api("missing", api_order={}, output_path=target)
    assert target.read_text(encoding="utf-8") == "existing reference"


def test_optional_docstring_formatter_and_empty_sections(
    generator: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = ModuleType("example_api")

    class Documented:
        """Original description."""

    Documented.__module__ = root.__name__
    Documented.__qualname__ = "Documented"
    root.Documented = Documented
    monkeypatch.setitem(sys.modules, root.__name__, root)
    target = tmp_path / "reference.rst"
    generator.generate_api(
        root.__name__,
        api_order={"Empty": [r"missing"], "Types": [r"example_api.Documented"]},
        output_path=target,
        format_docstring=lambda text: text.replace("Original", "Formatted"),
        skip_empty_sections=True,
    )
    text = target.read_text(encoding="utf-8")
    assert "Formatted description." in text
    assert "Empty\n" not in text
    assert "Miscellaneous\n" not in text
    assert not text.rstrip().endswith("----")
    assert Documented.__doc__ == "Original description."
