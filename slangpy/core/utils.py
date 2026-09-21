# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from os import PathLike, environ
import pathlib
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union, overload

from slangpy import (
    DeclReflection,
    ProgramLayout,
    TypeLayoutReflection,
    TypeReflection,
    DeviceType,
    PipelineCompilationMode,
    Device,
    DeviceDesc,
    NativeHandle,
    get_cuda_current_context_native_handles,
    BindlessDesc,
)
from slangpy.reflection import SlangType, SlangProgramLayout
import builtins

if TYPE_CHECKING:
    from slangpy import DeviceDescParam


@overload
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
) -> Device: ...


# Keep the descriptor overload last: Pyright uses it to suggest dictionary keys
# while the first positional argument is still being edited.
@overload
def create_device(desc: "DeviceDescParam") -> Device: ...


def create_device(*args: Any, **kwargs: Any) -> Device:
    """
    Create a device configured for SlangPy.

    Pass a :class:`DeviceDesc` or a descriptor dictionary, positionally or as
    ``desc``, for access to all device settings. Specify include paths through
    ``compiler_options.include_paths``. The descriptor is copied, and SlangPy's
    shader path is prepended without modifying caller-provided options.

    The original positional and keyword arguments remain supported. Descriptor
    calls cannot be combined with these legacy arguments.

    Both forms respect ``SLANGPY_DEVICE_TYPE_OVERRIDE`` and automatically set up
    Jupyter integration when applicable.

    Example::

        device = create_device({
            "type": DeviceType.vulkan,
            "compiler_options": {"include_paths": ["shaders"]},
            "enable_ray_tracing": False,
        })
    """
    if "desc" in kwargs or (args and isinstance(args[0], (DeviceDesc, dict))):
        return _create_device_from_desc(*args, **kwargs)
    return _create_device_legacy(*args, **kwargs)


def _resolve_device_type(type: DeviceType) -> DeviceType:
    # Allow overriding the device type for automatic testing.
    return {
        "d3d12": DeviceType.d3d12,
        "vulkan": DeviceType.vulkan,
        "cuda": DeviceType.cuda,
        "metal": DeviceType.metal,
    }.get(environ.get("SLANGPY_DEVICE_TYPE_OVERRIDE", "").lower(), type)


def _create_device_legacy(
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
    desc = DeviceDesc(
        {
            "type": type,
            "enable_debug_layers": enable_debug_layers,
            "compiler_options": {"include_paths": list(include_paths)},
            "enable_cuda_interop": enable_cuda_interop,
            "enable_print": enable_print,
            "enable_hot_reload": enable_hot_reload,
            "enable_compilation_reports": enable_compilation_reports,
            "pipeline_compilation_mode": pipeline_compilation_mode,
        }
    )
    if adapter_luid is not None:
        desc.adapter_luid = adapter_luid
    if existing_device_handles is not None:
        desc.existing_device_handles = existing_device_handles
    if bindless_options is not None:
        desc.bindless_options = bindless_options
    return _create_device_from_desc(desc)


def _create_device_from_desc(desc: "DeviceDescParam") -> Device:
    if not isinstance(desc, (DeviceDesc, dict)):
        raise TypeError("desc must be a DeviceDesc or a dictionary")
    desc = DeviceDesc(desc)
    desc.type = _resolve_device_type(desc.type)
    desc.compiler_options.include_paths = [
        str(pathlib.Path(__file__).parent.parent.absolute() / "slang")
    ] + list(desc.compiler_options.include_paths)
    device = Device(desc)

    if is_running_in_jupyter():
        # Import only when IPython is available and running in Jupyter.
        from slangpy.core.jupyter import setup_in_jupyter

        setup_in_jupyter(device)

    return device


@overload
def create_torch_device(
    type: DeviceType = DeviceType.automatic,
    torch_device: Any = None,
    enable_debug_layers: bool = False,
    include_paths: Sequence[Union[str, PathLike[str]]] = [],
    enable_print: bool = False,
    enable_hot_reload: bool = True,
    enable_compilation_reports: bool = False,
    pipeline_compilation_mode: PipelineCompilationMode = PipelineCompilationMode.serial,
) -> Device: ...


# As with create_device, keep the descriptor overload last for dictionary completion.
@overload
def create_torch_device(desc: "DeviceDescParam", torch_device: Any = None) -> Device: ...


def create_torch_device(*args: Any, **kwargs: Any) -> Device:
    """
    Helper to create a device configured properly for PyTorch integration. If device type is CUDA,
    slangpy will attempt to directly share the CUDA context with PyTorch. This is the recommended
    way of using SlangPy with PyTorch.

    If device type is not CUDA (eg d3d12, vulkan), this will create a device with cuda interop enabled,
    and rely on shared memory + semaphores to syncronize between SlangPy and PyTorch. This approach
    works, and is valuable if access to graphics features (such as a rasterizer) is critical, but hardware
    context switching and memcpys are expensive, resulting in substantially worse performance.

    Pass a :class:`DeviceDesc` or dictionary, positionally or as ``desc``, to access
    all device settings. Include paths belong in ``compiler_options.include_paths``.
    The optional ``torch_device`` selects the PyTorch device in either call form.
    Other legacy arguments cannot be combined with a descriptor.

    The descriptor is copied. ``existing_device_handles`` is replaced with the
    selected PyTorch context's handles, and ``enable_cuda_interop`` is set according
    to the device type after applying ``SLANGPY_DEVICE_TYPE_OVERRIDE``.

    Example::

        device = create_torch_device({
            "type": DeviceType.cuda,
            "compiler_options": {"include_paths": ["shaders"]},
        }, torch_device="cuda:0")
    """
    if "desc" in kwargs or (args and isinstance(args[0], (DeviceDesc, dict))):
        return _create_torch_device_from_desc(*args, **kwargs)
    return _create_torch_device_legacy(*args, **kwargs)


def _create_torch_device_legacy(
    type: DeviceType = DeviceType.automatic,
    torch_device: Any = None,
    enable_debug_layers: bool = False,
    include_paths: Sequence[Union[str, PathLike[str]]] = [],
    enable_print: bool = False,
    enable_hot_reload: bool = True,
    enable_compilation_reports: bool = False,
    pipeline_compilation_mode: PipelineCompilationMode = PipelineCompilationMode.serial,
) -> Device:
    return _create_torch_device_from_desc(
        {
            "type": type,
            "enable_debug_layers": enable_debug_layers,
            "compiler_options": {"include_paths": list(include_paths)},
            "enable_print": enable_print,
            "enable_hot_reload": enable_hot_reload,
            "enable_compilation_reports": enable_compilation_reports,
            "pipeline_compilation_mode": pipeline_compilation_mode,
        },
        torch_device,
    )


def _create_torch_device_from_desc(desc: "DeviceDescParam", torch_device: Any = None) -> Device:
    if not isinstance(desc, (DeviceDesc, dict)):
        raise TypeError("desc must be a DeviceDesc or a dictionary")
    desc = DeviceDesc(desc)
    desc.type = _resolve_device_type(desc.type)

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

    desc.existing_device_handles = handles
    desc.enable_cuda_interop = desc.type != DeviceType.cuda
    return create_device(desc)


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
