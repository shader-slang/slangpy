# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import hashlib
import os
import re
from typing import TYPE_CHECKING, Any, Optional

from slangpy.core.callsignature import generate_constants
from slangpy.core.enums import IOType
from slangpy.core.native import CallMode, CallDataMode, pack_arg, unpack_arg
from slangpy.core.calldata import _DUMP_SLANG_INTERMEDIATES, _DUMP_GENERATED_SHADERS

from slangpy import CommandEncoder, ShaderCursor, SlangLinkOptions, uint3, DeviceType
from slangpy.core.native import NativeCallRuntimeOptions
from slangpy.bindings.marshall import BindContext
from slangpy.bindings.boundvariable import BoundCall
from slangpy.bindings.boundvariableruntime import BoundCallRuntime
from slangpy.bindings.codegen import CodeGen

if TYPE_CHECKING:
    from slangpy.core.function import FunctionNode


class DispatchData:
    def __init__(self, func: "FunctionNode", **kwargs: dict[str, Any]) -> None:
        super().__init__()

        try:
            # Read temps from function.
            function = func
            build_info = function.calc_build_info()
            type_conformances = build_info.type_conformances or []
            session = build_info.module.session
            device = session.device
            thread_group_size = build_info.thread_group_size

            # Build 'unpacked' args (that handle IThis)
            unpacked_kwargs = {k: unpack_arg(v) for k, v in kwargs.items()}

            # Bind
            # Setup context
            # Determine call data mode based on device type
            call_data_mode = (
                CallDataMode.entry_point
                if build_info.module.device.info.type == DeviceType.cuda
                else CallDataMode.global_data
            )

            context = BindContext(
                func.module.layout,
                CallMode.prim,
                build_info.module.device_module,
                build_info.options,
                call_data_mode,
            )

            # Build the unbound signature from inputs and convert straight
            # to runtime data - don't need anything fancy for raw dispatch,
            # just type marshalls so data can be converted to dispatch args.
            bindings = BoundCall(context, **unpacked_kwargs)
            self.debug_only_bindings = bindings
            self.runtime = BoundCallRuntime(bindings)

            # Verify and get reflection data.
            if build_info.function.is_overloaded:
                raise ValueError("Cannot use raw dispatch on overloaded functions.")
            if build_info.this_type is not None:
                raise ValueError("Cannot use raw dispatch on type methods.")

            reflection = build_info.function.reflection
            slang_function = build_info.module.layout.find_function(reflection, None)

            # Ensure the function has no out or inout parameters, and no return value.
            if any([x.io_type != IOType.inn for x in slang_function.parameters]):
                raise ValueError("Raw dispatch functions cannot have out or inout parameters.")
            if slang_function.have_return_value:
                raise ValueError("Raw dispatch functions cannot have return values.")

            ep = None
            codegen = CodeGen()

            # Add constants
            generate_constants(build_info, codegen)

            # Try to load the entry point from the device module to see if there is an existing kernel.
            if any(
                [reflection.name == x.name for x in build_info.module.device_module.entry_points]
            ):
                ep = build_info.module.device_module.entry_point(reflection.name, type_conformances)

            # Due to current slang crash, fail if entry point exists and we're trying to set thread group size.
            if ep is not None and thread_group_size is not None:
                raise ValueError(
                    "Slang currently does not allow specifying thread_group_size for pre-existing kernels."
                )

            # Clear entry point if overriding thread group size
            if thread_group_size is not None:
                ep = None

            # If not a valid kernel, need to generate one.
            if ep is None:

                # Check parameter requirements for raw dispatch kernel gen.
                if len(reflection.parameters) < 1:
                    raise ValueError(
                        "To generate raw dispatch functions, first parameter must be a thread id."
                    )
                if (
                    not "uint" in slang_function.parameters[0].type.full_name
                    or slang_function.parameters[0].name != "dispatchThreadID"
                ):
                    raise ValueError(
                        f"To generate raw dispatch functions, first parameter must be named dispatchThreadID and uint1/2/3, current param is {slang_function.parameters[0].declaration}"
                    )

                # Generate params and arguments.
                params = ",".join(
                    [f"{slang_function.parameters[0].declaration}: SV_DispatchThreadID"]
                    + ["uniform " + x.declaration for x in slang_function.parameters[1:]]
                )
                args = ",".join([x.name for x in slang_function.parameters])

                if thread_group_size is None:
                    thread_group_size = uint3(32, 1, 1)

                # Generate mini-kernel for calling the function.
                codegen.kernel.append_code(
                    f"""
import "{build_info.module.device_module.name}";
[shader("compute")]
[numthreads({thread_group_size.x}, {thread_group_size.y}, {thread_group_size.z})]
void {reflection.name}_entrypoint({params}) {{
    {reflection.name}({args});
}}
"""
                )

            # Add imports
            codegen.add_import("slangpy")

            # Generate code
            code = codegen.finish(
                call_data=True,
                input_load_store=True,
                header=True,
                kernel=True,
                imports=True,
                trampoline=True,
                context=True,
                snippets=True,
                call_data_structs=True,
                constants=True,
                use_param_block_for_call_data=context.device.info.type != DeviceType.cuda,
            )

            sanitized = ""
            if _DUMP_GENERATED_SHADERS or _DUMP_SLANG_INTERMEDIATES:
                # Write the shader to a file for debugging.
                os.makedirs(".temp", exist_ok=True)
                santized_module = re.sub(r"[<>, ./]", "_", build_info.module.name)
                sanitized = re.sub(r"[<>, ]", "_", build_info.name)
                fn = f".temp/{santized_module}_{sanitized}_dispatch.slang"
                with open(
                    fn,
                    "w",
                ) as f:
                    f.write(code)

            # Hash the code to get a unique identifier for the module.
            # We add type conformances to the start of the code to ensure that the hash is unique
            code_minus_header = (
                "[DispatchData]\n" + str(build_info.type_conformances) + code[len(codegen.header) :]
            )
            hash = hashlib.sha256(code_minus_header.encode()).hexdigest()

            # Check if we've already built this module.
            if hash in build_info.module.compute_pipeline_cache:
                # Get pipeline from cache if we have
                self.compute_pipeline = build_info.module.compute_pipeline_cache[hash]
                self.device = build_info.module.device
            else:
                # Load the module
                module = session.load_module_from_source(
                    hashlib.sha256(code.encode()).hexdigest()[0:16], code
                )

                # Get entry point if one wasn't specified
                if ep is None:
                    ep = module.entry_point(f"{reflection.name}_entrypoint", type_conformances)

                # Link the program
                opts = SlangLinkOptions()
                opts.dump_intermediates = _DUMP_SLANG_INTERMEDIATES
                opts.dump_intermediates_prefix = sanitized
                program = session.link_program(
                    [module, build_info.module.device_module] + build_info.module.link,
                    [ep],
                    opts,
                )
                self.compute_pipeline = device.create_compute_pipeline(
                    program, defer_target_compilation=True
                )
                self.device = device

        except Exception as e:
            raise e

    def dispatch(
        self,
        opts: "NativeCallRuntimeOptions",
        thread_count: uint3,
        vars: dict[str, Any] = {},
        command_encoder: Optional[CommandEncoder] = None,
        **kwargs: dict[str, Any],
    ) -> None:

        # Merge uniforms
        uniforms: dict[str, Any] = {}
        if opts.uniforms is not None:
            for u in opts.uniforms:
                if isinstance(u, dict):
                    uniforms.update(u)
                else:
                    uniforms.update(u(self))  # type: ignore (need to work out native dispatch)
        uniforms.update(vars)

        # Build 'unpacked' args (that handle IThis)
        unpacked_kwargs = {k: unpack_arg(v) for k, v in kwargs.items()}

        # Use python marshalls to convert provided arguments to dispatch args.
        call_data = {}
        self.runtime.write_raw_dispatch_data(call_data, unpacked_kwargs)

        # Create temporary command encoder if none is provided
        temp_command_encoder: Optional[CommandEncoder] = None
        if command_encoder is None:
            temp_command_encoder = self.device.create_command_encoder()
            command_encoder = temp_command_encoder

        # Dispatch kernel
        compute_pass = command_encoder.begin_compute_pass()
        cursor = ShaderCursor(compute_pass.bind_pipeline(self.compute_pipeline))
        cursor.write(uniforms)
        cursor.find_entry_point(0).write(call_data)
        compute_pass.dispatch(thread_count)
        compute_pass.end()

        # Submit if we created a temporary command encoder
        if temp_command_encoder is not None:
            self.device.submit_command_buffer(temp_command_encoder.finish())
            command_encoder = None

        # If just adding to command encoder, post dispatch is redundant
        if command_encoder is not None:
            return

        # Push updated 'this' values back to original objects
        for k, arg in kwargs.items():
            pack_arg(arg, unpacked_kwargs[k])
