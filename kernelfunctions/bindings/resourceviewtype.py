

from typing import Any
from kernelfunctions.backend import ResourceView, Texture, ResourceViewType, ResourceUsage
from kernelfunctions.bindings.texturetype import get_or_create_python_texture_type
from kernelfunctions.typeregistry import PYTHON_SIGNATURES, PYTHON_TYPES


def _get_or_create_python_type(value: Any):
    assert isinstance(value, ResourceView)
    if isinstance(value.resource, Texture):
        if value.type == ResourceViewType.shader_resource:
            return get_or_create_python_texture_type(value.resource, ResourceUsage.shader_resource)
        elif value.type == ResourceViewType.unordered_access:
            return get_or_create_python_texture_type(value.resource, ResourceUsage.unordered_access)
        else:
            raise ValueError(f"Unsupported resource view type {value.type}")
    else:
        raise ValueError(f"Unsupported resource view resource {value.resource}")


def _get_signature(value: Any):
    assert isinstance(value, ResourceView)
    if isinstance(value.resource, Texture):
        x = value.resource
        return f"[texture,{x.desc.type},{value.type},{x.desc.format},{x.array_size>1}]"
    else:
        raise ValueError(f"Unsupported resource view resource {value.resource}")


PYTHON_TYPES[ResourceView] = _get_or_create_python_type
PYTHON_SIGNATURES[ResourceView] = _get_signature
