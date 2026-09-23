# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from typing import Any, Optional

import numpy as np

from slangpy import Device, TypeLayoutReflection, TypeReflection
from slangpy.bindings.marshall import Marshall
from slangpy.bindings.typeregistry import get_or_create_type
from slangpy.native_func import BaseStruct
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


def _same_path(a: str, b: str) -> bool:
    """True if `a` and `b` name the same location, using inode identity so that
    symlinks and case-insensitive filesystems compare equal, with a lexical
    fallback when either path does not exist on disk."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def _slang_session_has_include_path(device: Device, path: str) -> Optional[bool]:
    """Return whether `path` is present in the device's default Slang session's
    configured include paths, or None if they cannot be read. (This inspects the
    explicit compiler_options.include_paths, not Slang's implicit default search
    paths -- correct here because SHADER_PATH is never a Slang system default.)
    The builtin 'slangpy' module is resolved from this session, so a missing
    SlangPy shader path here is what turns a Tensor.from_numpy call into
    "cannot open file 'slangpy.slang'". None means "unknown" and callers must
    not treat it as absent."""
    try:
        include_paths = device.slang_session.desc.compiler_options.include_paths
    except Exception:
        return None
    return any(_same_path(str(p), path) for p in include_paths)


def _get_lookup_module(device: Device) -> SlangProgramLayout:
    try:
        return get_builtin_layout(device)
    except Exception as e:
        # Rewrite the error into an actionable hint only when 'slangpy.slang'
        # itself could not be opened and the shader path is confirmed absent;
        # any other failure (present/unknown path, unrelated error, or a compile
        # error in a resolvable module) is re-raised unchanged. Broad except also
        # covers macOS, where the diagnostic arrives as RuntimeError; SHADER_PATH
        # is imported lazily because it is defined late in slangpy/__init__.py.
        from slangpy import SHADER_PATH, SlangCompileError

        text = str(e)
        slangpy_file_unresolved = "cannot open file 'slangpy.slang'" in text
        if not (
            slangpy_file_unresolved
            and _slang_session_has_include_path(device, SHADER_PATH) is False
        ):
            raise
        raise SlangCompileError(
            "Could not load SlangPy's builtin 'slangpy' shader module: the device "
            "was created without SlangPy's shader include path, so 'slangpy.slang' "
            "cannot be resolved. Create the device with spy.create_device() "
            "(recommended), or add spy.SHADER_PATH to compiler_options['include_paths'] "
            f"when constructing spy.Device().\n\nOriginal error: {e}"
        ) from e


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

    if isinstance(element_type, (SlangType, BaseStruct)):
        return native_resolve_layout(device, element_type, None)

    return _get_lookup_module(device)


def resolve_element_type(program_layout: SlangProgramLayout, element_type: Any) -> SlangType:
    """
    Resolve a container element type from strings, Python values, structs, or reflection objects.
    """
    if isinstance(element_type, Marshall):
        element_type = element_type.slang_type

    if isinstance(element_type, (SlangType, TypeReflection, TypeLayoutReflection, BaseStruct, str)):
        resolved = native_resolve_element_type(program_layout, element_type)
    else:
        resolved = get_or_create_type(program_layout, element_type).slang_type

    if resolved is None:
        raise ValueError("Element type could not be resolved")
    return resolved
