# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


PROJECT_DIR = Path(__file__).resolve().parents[3]


def load_docs_tool() -> ModuleType:
    """Load tools/docs.py without making tools a Python package."""
    module_name = "slangpy_docs_tool"
    spec = importlib.util.spec_from_file_location(module_name, PROJECT_DIR / "tools" / "docs.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


DOCS_TOOL = load_docs_tool()


def test_generated_api_check_accepts_stable_values(tmp_path: Path) -> None:
    path = tmp_path / "api.rst"
    path.write_text(
        ".. py:data:: slangpy.SGL_VERSION\n    :type: str\n    :value: \"0.43.0\"\n",
        encoding="utf-8",
    )

    DOCS_TOOL.check_generated_api(path)


@pytest.mark.parametrize(
    "unsafe_text",
    [
        ".. py:data:: slangpy.SGL_GIT_VERSION\n    :value: \"commit: abc / branch: main\"\n",
        ".. py:data:: slangpy.SHADER_PATH\n    :value: \"C:\\work\\slangpy\"\n",
        ".. py:data:: value\n    :value: \"/home/user/slangpy\"\n",
        ".. py:data:: value\n    :value: \"build/windows-msvc/Release\"\n",
        ".. py:data:: slangpy.UNREVIEWED_CONSTANT\n    :value: 42\n",
        ".. py:attribute:: method\n    :value: <function method at 0x12345678>\n",
        ".. py:method:: f(value: object = <object at 0x12345678>) -> None\n",
        ".. py:data:: value\n    :value: <built-in function warn>\n",
    ],
)
def test_generated_api_check_rejects_unstable_values(
    tmp_path: Path, unsafe_text: str
) -> None:
    path = tmp_path / "api.rst"
    path.write_text(unsafe_text, encoding="utf-8")

    with pytest.raises(DOCS_TOOL.DocumentationError, match="not reproducible"):
        DOCS_TOOL.check_generated_api(path)


def test_generated_api_check_rejects_missing_snapshot(tmp_path: Path) -> None:
    with pytest.raises(DOCS_TOOL.DocumentationError, match="does not exist"):
        DOCS_TOOL.check_generated_api(tmp_path / "missing.rst")


def test_prepare_tutorials_replaces_stale_output(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    output_dir.mkdir()
    (source_dir / "tutorial.ipynb").write_text("current", encoding="utf-8")
    (output_dir / "stale.txt").write_text("stale", encoding="utf-8")

    DOCS_TOOL.prepare_tutorials(source_dir, output_dir)

    assert (output_dir / "tutorial.ipynb").read_text(encoding="utf-8") == "current"
    assert not (output_dir / "stale.txt").exists()
