from typing import Any
from slangpy.core.enums import PrimType, IOType
from slangpy.core.native import Shape, CallContext, AccessType, TypeReflection
from slangpy.backend import ResourceUsage, Buffer, ResourceView, ResourceViewType, Texture, ResourceType, FormatType, get_format_info
from slangpy.bindings import CodeGenBlock, BindContext, ReturnContext, BaseType, BaseTypeImpl, BoundVariable, PYTHON_TYPES, PYTHON_SIGNATURES, BoundVariableRuntime, get_or_create_type
from slangpy.reflection import SlangProgramLayout, SlangType, TYPE_OVERRIDES, is_matching_array_type
from slangpy.types import NDBuffer, NDDifferentiableBuffer
import slangpy.bindings.typeregistry as tr
import slangpy.reflection as kfr
from slangpy.builtin.texturetype import get_or_create_python_texture_type


def _get_or_create_python_type(layout: SlangProgramLayout, value: Any):
    assert isinstance(value, ResourceView)
    if isinstance(value.resource, Texture):
        if value.type == ResourceViewType.shader_resource:
            return get_or_create_python_texture_type(layout, value.resource, ResourceUsage.shader_resource)
        elif value.type == ResourceViewType.unordered_access:
            return get_or_create_python_texture_type(layout, value.resource, ResourceUsage.unordered_access)
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