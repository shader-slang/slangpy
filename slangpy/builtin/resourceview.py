# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Any

from slangpy import TextureView
from slangpy.bindings import PYTHON_SIGNATURES, PYTHON_TYPES
from slangpy.builtin.texture import get_or_create_python_texture_type
from slangpy.reflection import SlangProgramLayout


def _get_or_create_python_type(layout: SlangProgramLayout, value: Any):
    assert isinstance(value, TextureView)
    desc = value.texture.desc
    return get_or_create_python_texture_type(
        layout, desc.format, desc.type, desc.usage, desc.array_length, desc.sample_count
    )


def _get_signature(value: Any):
    assert isinstance(value, TextureView)
    x = value.texture
    return f"[texture,{x.desc.type},{x.desc.usage},{x.desc.format}]"


PYTHON_TYPES[TextureView] = _get_or_create_python_type
PYTHON_SIGNATURES[TextureView] = _get_signature
