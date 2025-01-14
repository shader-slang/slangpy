# SPDX-License-Identifier: Apache-2.0
from os import PathLike
import pathlib
from typing import Sequence, Union

from slangpy.backend import (DeclReflection, ProgramLayout,
                             TypeLayoutReflection, TypeReflection,
                             DeviceType, Device)


def create_device(type: DeviceType = DeviceType.automatic, enable_debug_layers: bool = False, adapter_luid: Sequence[int] | None = None, include_paths: Sequence[str | PathLike] = []):
    """
    Create an SGL device with basic settings for SlangPy. For full control over device init, 
    use sgl.create_device directly, being sure to add slangpy.SHADER_PATH
    to the list of include paths for the compiler.
    """

    shaderpath = str(pathlib.Path(__file__).parent.parent.absolute() / "slang")

    return Device(
        type=type,
        compiler_options={
            "include_paths": [
                shaderpath,
            ]+list(include_paths),
        },
        enable_cuda_interop=True,
        enable_debug_layers=enable_debug_layers,
        adapter_luid=adapter_luid)


def find_type_layout_for_buffer(program_layout: ProgramLayout, slang_type: Union[str, TypeReflection, TypeLayoutReflection]):
    if isinstance(slang_type, str):
        slang_type_name = slang_type
    elif isinstance(slang_type, (TypeReflection, TypeLayoutReflection)):
        slang_type_name = slang_type.name
    buffer_type = program_layout.find_type_by_name(f"StructuredBuffer<{slang_type_name}>")
    buffer_layout = program_layout.get_type_layout(buffer_type)
    return buffer_layout.element_type_layout


def try_find_type_decl(root: DeclReflection, type_name: str):

    type_names = type_name.split("::")

    type_decl = root
    while len(type_names) > 0:
        type_name = type_names.pop(0)
        type_decl = type_decl.find_first_child_of_kind(
            DeclReflection.Kind.struct, type_name)
        if type_decl is None:
            return None

    return type_decl


def try_find_type_via_ast(root: DeclReflection, type_name: str):
    type_decl = try_find_type_decl(root, type_name)
    return type_decl.as_type() if type_decl is not None else None


def try_find_function_overloads_via_ast(root: DeclReflection, type_name: str, func_name: str):

    type_decl = try_find_type_decl(root, type_name)
    if type_decl is None:
        return (None, None)

    func_decls = type_decl.find_children_of_kind(DeclReflection.Kind.func, func_name)
    return (type_decl.as_type(), [x.as_function() for x in func_decls])


def parse_generic_signature(name: str):
    # Find start of generic arguments, return name if not found
    argument_start = name.find("<")
    if argument_start == -1:
        return (name, [])

    type_name = name[:argument_start].strip()

    # Read full argument names, using depth check to avoid recursion
    depth = 0
    args = []
    argument_start += 1
    pos = argument_start
    while pos < len(name):
        if name[pos] == "<":
            depth += 1
        elif name[pos] == ">":
            depth -= 1
        elif name[pos] == ',' and depth == 0:
            args.append(name[argument_start:pos].strip())
            argument_start = pos+1
        pos += 1
    args.append(name[argument_start:pos-1].strip())

    return (type_name, args)


def parse_generic_signature_tree(name: str):
    res = _recurse_parse_generic_signature(name, 0, len(name))
    assert res[0] == len(name)
    return res[1:]


def _recurse_parse_generic_signature(name: str, start: int, end: int):

    # Find start of generic arguments
    argument_start = name.find("<", start)
    if argument_start == -1:
        # No generic, so arg will be terminated either by a comma or end of args
        comma = name.find(",", start, end)
        if comma == -1:
            return (end, name[start:end].strip(), [])
        else:
            return (comma, name[start:comma].strip(), [])

    # Got a generic, so get the name of the type (or value if it's a value argument)
    type_name = name[start:argument_start].strip()

    # Step past the '<'
    argument_start += 1

    # Find the end of the generic arguments
    gend = name.rfind(">", argument_start, end)

    # Parse the arguments until reached end
    args = []
    while argument_start < gend:
        # Recurse, which returns the end of the argument that was read, along with name and children
        arg_end, arg, subargs = _recurse_parse_generic_signature(
            name, argument_start, gend)
        args.append((arg, subargs))

        # Step past the end of last argument (would have been comma or end of string)
        argument_start = arg_end+1

    # Return the end of the generic arguments, along with the name and children
    return (gend+1, type_name, args)


def shape_to_contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(shape) == 0:
        return ()

    strides = (1, )
    for dim in reversed(shape[1:]):
        strides = (dim * strides[0], ) + strides

    return strides
