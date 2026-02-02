# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Optional
from slangpy import Device, TypeReflection, TypeLayoutReflection
from slangpy.bindings.marshall import Marshall
from slangpy.reflection import SlangProgramLayout, SlangType, ScalarType
from slangpy.reflection.reflectiontypes import scalar_names
from slangpy.bindings.typeregistry import get_or_create_type
from slangpy.core.struct import Struct

import numpy as np

_global_lookup_modules: dict[Device, SlangProgramLayout] = {}

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


def _on_device_close(device: Device):
    del _global_lookup_modules[device]


def _load_lookup_module(device: Device):
    dummy_module = device.load_module_from_source("slangpy_layout", 'import "slangpy";')
    _global_lookup_modules[device] = SlangProgramLayout(dummy_module.layout)


def _hot_reload_lookup_module(device: Device):
    if device in _global_lookup_modules:
        dummy_module = device.load_module_from_source("slangpy_layout", 'import "slangpy";')
        _global_lookup_modules[device].on_hot_reload(dummy_module.layout)


def _get_lookup_module(device: Device) -> SlangProgramLayout:
    if device not in _global_lookup_modules:
        _load_lookup_module(device)
        device.register_device_close_callback(_on_device_close)
        device.register_shader_hot_reload_callback(lambda _: _hot_reload_lookup_module(device))

    return _global_lookup_modules[device]


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
    Attempt to find a SlangProgramLayout that can be used for further type lookups.
    If program_layout is provided, it is returned directly.
    Otherwise, if element_type is a SlangType, Marshall, or Struct, the program layout is taken
    from there.
    If none of those apply, the global lookup module for the device is returned.
    """
    if program_layout is None:
        if isinstance(element_type, SlangType):
            program_layout = element_type.program
        elif isinstance(element_type, Marshall):
            program_layout = element_type.slang_type.program
        elif isinstance(element_type, Struct):
            program_layout = element_type.module.layout
        else:
            program_layout = _get_lookup_module(device)
    return program_layout


def resolve_element_type(program_layout: SlangProgramLayout, element_type: Any) -> SlangType:
    """
    Attempt to resolve an element type for a container (which can be specified in a variety of ways,
    such as strings, python types or explicit slang types), and resolve to a specific
    slang reflection type.
    """
    if isinstance(element_type, SlangType):
        pass
    elif isinstance(element_type, str):
        element_type = program_layout.find_type_by_name(element_type)
    elif isinstance(element_type, Struct):
        if element_type.module.layout == program_layout:
            element_type = element_type.struct
        else:
            element_type = program_layout.find_type_by_name(element_type.full_name)
    elif isinstance(element_type, TypeReflection):
        element_type = program_layout.find_type_by_name(element_type.full_name)
    elif isinstance(element_type, TypeLayoutReflection):
        element_type = program_layout.find_type_by_name(element_type.type.full_name)
    elif isinstance(element_type, Marshall):
        if element_type.slang_type.program == program_layout:
            element_type = element_type.slang_type
        else:
            element_type = program_layout.find_type_by_name(element_type.slang_type.full_name)
    else:
        bt = get_or_create_type(program_layout, element_type)
        element_type = bt.slang_type
    if element_type is None:
        raise ValueError("Element type could not be resolved")
    return element_type
