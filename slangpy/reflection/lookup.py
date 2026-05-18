# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Optional

import numpy as np

from slangpy import Device, TypeReflection
from slangpy.bindings.marshall import Marshall
from slangpy.bindings.typeregistry import get_or_create_type
from slangpy.native_refl import get_builtin_layout
from slangpy.native_refl import resolve_element_type as native_resolve_element_type
from slangpy.native_refl import resolve_layout as native_resolve_layout
from slangpy.reflection import ScalarType, SlangProgramLayout, SlangType
from slangpy.reflection.reflectiontypes import scalar_names

ST = TypeReflection.ScalarType
_numpy_to_sgl = {
    "int8": ST.int8,
    "int16": ST.int16,
    "int32": ST.int32,
    "int64": ST.int64,
    "uint8": ST.uint8,
    "uint16": ST.uint16,
    "uint32": ST.uint32,
    "uint64": ST.uint64,
    "float16": ST.float16,
    "float32": ST.float32,
    "float64": ST.float64,
}
_sgl_to_numpy = {y: x for x, y in _numpy_to_sgl.items()}


def _get_lookup_module(device: Device) -> SlangProgramLayout:
    return get_builtin_layout(device)


def innermost_type(slang_type: SlangType) -> SlangType:
    while True:
        if slang_type.element_type is not None and slang_type.element_type is not slang_type:
            slang_type = slang_type.element_type
        else:
            return slang_type


def slang_to_numpy(slang_dtype: SlangType):
    """
    Convert a Slang reflection type to a NumPy dtype. If the slang type is container (eg array or
    vector), the innermost element type is used.
    """
    elem_type = innermost_type(slang_dtype)
    if isinstance(elem_type, ScalarType) and elem_type.slang_scalar_type in _sgl_to_numpy:
        return np.dtype(_sgl_to_numpy[elem_type.slang_scalar_type])
    return None


def numpy_to_slang(
    np_dtype: np.dtype[Any], device: Device, program_layout: Optional[SlangProgramLayout]
) -> Optional[SlangType]:
    """
    Convert a NumPy dtype to a Slang reflection type.
    """
    name = np_dtype.base.name
    if name not in _numpy_to_sgl:
        return None
    slang_dtype = scalar_names[_numpy_to_sgl[name]]
    if np_dtype.ndim > 0:
        for dim in reversed(np_dtype.shape):
            slang_dtype += f"[{dim}]"

    if program_layout is None:
        program_layout = _get_lookup_module(device)
    return program_layout.find_type_by_name(slang_dtype)


def resolve_program_layout(
    device: Device, element_type: Any, program_layout: Optional[SlangProgramLayout]
) -> SlangProgramLayout:
    """
    Find a native reflection layout for further type lookups.
    """
    if program_layout is not None:
        return program_layout

    if isinstance(element_type, Marshall):
        element_type = element_type.slang_type

    return native_resolve_layout(device, element_type, None)


def resolve_element_type(program_layout: SlangProgramLayout, element_type: Any) -> SlangType:
    """
    Resolve a container element type from strings, Python values, structs, or reflection objects.
    """
    if isinstance(element_type, Marshall):
        element_type = element_type.slang_type

    resolved = native_resolve_element_type(program_layout, element_type)
    if resolved is None:
        resolved = get_or_create_type(program_layout, element_type).slang_type

    if resolved is None:
        raise ValueError("Element type could not be resolved")
    return resolved
