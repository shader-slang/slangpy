# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from slangpy import DeviceType, ProgramLayout, TypeReflection


class SlangPyError(ValueError):
    """
    Raised for SlangPy usage the functional API rejects up front because it cannot produce a
    correct result.
    """


def check_no_module_push_constants(
    device_type: DeviceType, function_name: str, layout: ProgramLayout
) -> None:
    """
    On Vulkan, raise if the linked program has a global push-constant block.

    The generated kernel's call data is itself a push-constant block (entry-point uniforms) or a
    descriptor set (ParameterBlock<CallData>), and neither currently coexists with a module-level
    ``[[vk::push_constant]]``: calls silently read the wrong bytes.
    See https://github.com/shader-slang/slangpy/issues/1232.

    :param device_type: Type of the device the program is built for.
    :param function_name: Name of the called function, for the error message.
    :param layout: Layout of the linked program, which includes globals of imported modules.
    """
    if device_type != DeviceType.vulkan:
        return
    push_constant = TypeReflection.ParameterCategory.push_constant_buffer
    for param in layout.parameters:
        if param.type_layout.has_category(push_constant):
            raise SlangPyError(
                f"Could not call function '{function_name}': module global push constant "
                f"'{param.name}' is not supported by the functional API on Vulkan. Remove "
                f"[[vk::push_constant]] so it becomes an ordinary uniform, and set it with "
                f"function.set({param.name}=...)."
            )
