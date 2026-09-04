# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from os import PathLike, environ
import pathlib
from typing import Any, Optional, Sequence, Union

from slangpy import (
    DeclReflection,
    ProgramLayout,
    TypeLayoutReflection,
    TypeReflection,
    DeviceType,
    PipelineCompilationMode,
    Device,
    NativeHandle,
    get_cuda_current_context_native_handles,
    BindlessDesc,
)
from slangpy.reflection import SlangType, SlangProgramLayout
import builtins


def create_device(
    type: DeviceType = DeviceType.automatic,
    enable_debug_layers: bool = False,
    adapter_luid: Optional[Sequence[int]] = None,
    include_paths: Sequence[Union[str, PathLike[str]]] = [],
    enable_cuda_interop: bool = False,
    enable_print: bool = False,
    enable_hot_reload: bool = True,
    enable_compilation_reports: bool = False,
    pipeline_compilation_mode: PipelineCompilationMode = PipelineCompilationMode.serial,
    existing_device_handles: Optional[Sequence[NativeHandle]] = None,
    bindless_options: Optional[BindlessDesc] = None,
) -> Device:
    """
    Create a GPU device configured for SlangPy's functional API.

    The built-in SlangPy shader library is prepended to ``include_paths``. The
    ``SLANGPY_DEVICE_TYPE_OVERRIDE`` environment variable can override ``type``
    with ``d3d12``, ``vulkan``, ``cuda``, or ``metal``. In a Jupyter kernel, the
    returned device is also registered with SlangPy's notebook integration.

    :param type: Backend to create, or ``automatic`` to select an available backend.
    :param enable_debug_layers: Enable backend validation and debug layers.
    :param adapter_luid: Optional adapter LUID as the two 32-bit integer components
        used on Windows.
    :param include_paths: Additional search paths for Slang modules and includes.
    :param enable_cuda_interop: Enable CUDA sharing support on a graphics backend.
    :param enable_print: Enable Slang ``print`` output from GPU programs.
    :param enable_hot_reload: Watch loaded shader modules for changes.
    :param enable_compilation_reports: Retain shader compilation reports on the device.
    :param pipeline_compilation_mode: Select serial or asynchronous pipeline compilation.
    :param existing_device_handles: Native handles used to wrap an existing device or
        CUDA context.
    :param bindless_options: Optional bindless descriptor configuration.
    :return: The configured :class:`slangpy.Device`.

    Example:

    .. code-block:: python

        import slangpy as spy

        device = spy.create_device(enable_debug_layers=True)

    For complete control over initialization, construct :class:`slangpy.Device`
    directly and include ``slangpy.SHADER_PATH`` in its compiler search paths.
    """

    shaderpath = str(pathlib.Path(__file__).parent.parent.absolute() / "slang")

    # Allow overriding the device type via environment variable (for automatic testing).
    if "SLANGPY_DEVICE_TYPE_OVERRIDE" in environ:
        type = {
            "d3d12": DeviceType.d3d12,
            "vulkan": DeviceType.vulkan,
            "cuda": DeviceType.cuda,
            "metal": DeviceType.metal,
        }.get(environ["SLANGPY_DEVICE_TYPE_OVERRIDE"].lower(), type)

    device = Device(
        type=type,
        compiler_options={
            "include_paths": [
                shaderpath,
            ]
            + list(include_paths),
        },
        enable_debug_layers=enable_debug_layers,
        adapter_luid=adapter_luid,
        enable_cuda_interop=enable_cuda_interop,
        enable_print=enable_print,
        enable_hot_reload=enable_hot_reload,
        enable_compilation_reports=enable_compilation_reports,
        pipeline_compilation_mode=pipeline_compilation_mode,
        existing_device_handles=existing_device_handles,
        bindless_options=bindless_options,
    )

    if is_running_in_jupyter():
        # Don't import until we know we're running in jupyter and we are certain the IPython module is available
        from slangpy.core.jupyter import setup_in_jupyter

        setup_in_jupyter(device)

    return device


def create_torch_device(
    type: DeviceType = DeviceType.automatic,
    torch_device: Any = None,
    enable_debug_layers: bool = False,
    include_paths: Sequence[Union[str, PathLike[str]]] = [],
    enable_print: bool = False,
    enable_hot_reload: bool = True,
    enable_compilation_reports: bool = False,
    pipeline_compilation_mode: PipelineCompilationMode = PipelineCompilationMode.serial,
) -> Device:
    """
    Create a SlangPy device that interoperates with the active PyTorch CUDA device.

    A CUDA SlangPy device reuses PyTorch's current CUDA context. Other backends are
    created with CUDA interop enabled and synchronize through shared resources; that
    path permits graphics features but can incur context switches and copies.

    :param type: SlangPy backend to create.
    :param torch_device: PyTorch CUDA device or device index. The current device is
        used when omitted.
    :param enable_debug_layers: Enable backend validation and debug layers.
    :param include_paths: Additional Slang module and include search paths.
    :param enable_print: Enable Slang ``print`` output from GPU programs.
    :param enable_hot_reload: Watch loaded shader modules for changes.
    :param enable_compilation_reports: Retain shader compilation reports on the device.
    :param pipeline_compilation_mode: Select serial or asynchronous pipeline compilation.
    :return: A device sharing or interoperating with PyTorch's CUDA context.
    :raises ImportError: If PyTorch is not installed.

    Example:

    .. code-block:: python

        import slangpy as spy

        device = spy.create_torch_device(spy.DeviceType.cuda)
    """

    # Import and init torch
    import torch

    # Ensure torch cuda is initialized
    torch.cuda.init()

    # These lines ensure that torch has set a default context
    torch.cuda.current_device()
    torch.cuda.current_stream()

    # Use current device if not specified
    if torch_device is None:
        torch_device = torch.cuda.current_device()

    # Switch to the correct device then read cuda context
    with torch.device(torch_device):
        handles = get_cuda_current_context_native_handles()

    return create_device(
        type=type,
        enable_debug_layers=enable_debug_layers,
        include_paths=include_paths,
        enable_cuda_interop=type != DeviceType.cuda,
        enable_print=enable_print,
        enable_hot_reload=enable_hot_reload,
        enable_compilation_reports=enable_compilation_reports,
        pipeline_compilation_mode=pipeline_compilation_mode,
        existing_device_handles=handles,
    )


def find_type_layout_for_buffer(
    program_layout: ProgramLayout,
    slang_type: Union[str, TypeReflection, TypeLayoutReflection],
):
    if isinstance(slang_type, str):
        slang_type_name = slang_type
    elif isinstance(slang_type, TypeReflection):
        slang_type_name = slang_type.full_name
    elif isinstance(slang_type, TypeLayoutReflection):
        slang_type_name = slang_type.type.full_name
    buffer_type = program_layout.find_type_by_name(f"StructuredBuffer<{slang_type_name}>")
    buffer_layout = program_layout.get_type_layout(buffer_type)
    return buffer_layout.element_type_layout


def try_find_type_decl(root: DeclReflection, type_name: str):

    type_names = type_name.split("::")

    type_decl = root
    while len(type_names) > 0:
        type_name = type_names.pop(0)
        type_decl = type_decl.find_first_child_of_kind(DeclReflection.Kind.struct, type_name)
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


# This function checks if we can replace a python value's type with the destination
# type in resolve_type. This lets us e.g. pass a python int to a python uint16_t
# This is done by specializing a class with a specific generic constraint.
# If specialization succeeds, slang tells us this is A-OK
def is_type_castable_on_host(
    from_type: Union[SlangType, str],
    to_type: Union[SlangType, str],
    program: Optional[SlangProgramLayout] = None,
) -> bool:
    if program is None:
        if isinstance(from_type, SlangType):
            program = from_type.program
        elif isinstance(to_type, SlangType):
            program = to_type.program
    if isinstance(from_type, SlangType):
        from_type = from_type.full_name
    if isinstance(to_type, SlangType):
        to_type = to_type.full_name
    if program is None:
        raise ValueError("Program must be provided or inferable from from_type or to_type")
    witness_name = f"impl::AllowedConversionWitness<{from_type}, {to_type}>"
    witness = program.find_type_by_name(witness_name)
    return witness is not None


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
        elif name[pos] == "," and depth == 0:
            args.append(name[argument_start:pos].strip())
            argument_start = pos + 1
        pos += 1
    args.append(name[argument_start : pos - 1].strip())

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
        arg_end, arg, subargs = _recurse_parse_generic_signature(name, argument_start, gend)
        args.append((arg, subargs))

        # Step past the end of last argument (would have been comma or end of string)
        argument_start = arg_end + 1

    # Return the end of the generic arguments, along with the name and children
    return (gend + 1, type_name, args)


def shape_to_contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(shape) == 0:
        return ()

    strides = (1,)
    for dim in reversed(shape[1:]):
        strides = (dim * strides[0],) + strides

    return strides


def is_running_in_jupyter():
    # Jupyter will inject the get_ipython() function into the globals. First
    # check it is available there before calling it
    if hasattr(builtins, "get_ipython"):
        shell = get_ipython().__class__.__name__  # type: ignore
        return shell == "ZMQInteractiveShell"
    else:
        return False
