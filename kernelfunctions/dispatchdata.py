
import hashlib
import os
import re
from typing import TYPE_CHECKING, Any

from kernelfunctions.backend import CommandBuffer, SlangLinkOptions, uint3
from kernelfunctions.core.enums import IOType
from kernelfunctions.core.native import CallMode, unpack_arg, pack_arg
from kernelfunctions.core.basetype import BindContext
from kernelfunctions.core.boundvariable import BoundCall
from kernelfunctions.core.boundvariableruntime import BoundCallRuntime


if TYPE_CHECKING:
    from kernelfunctions.function import Function


class DispatchData:
    def __init__(
        self,
        func: "Function",
        **kwargs: dict[str, Any]
    ) -> None:
        super().__init__()

        try:
            # Read temps from function.
            function = func
            type_conformances = function._type_conformances or []
            session = function.module.session
            device = session.device
            thread_group_size = function._thread_group_size

            # Build 'unpacked' args (that handle IThis)
            unpacked_kwargs = {k: unpack_arg(v) for k, v in kwargs.items()}

            # Bind
            # Setup context
            context = BindContext(func.module.layout, CallMode.prim,
                                  function.module.device_module, function._options or {})

            # Build the unbound signature from inputs and convert straight
            # to runtime data - don't need anything fancy for raw dispatch,
            # just type marshalls so data can be converted to dispatch args.
            bindings = BoundCall(context, **unpacked_kwargs)
            self.debug_only_bindings = bindings
            self.runtime = BoundCallRuntime(bindings)

            # Verify and get reflection data.
            if len(function.reflections) > 1 or function.reflections[0].is_overloaded:
                raise ValueError("Cannot use raw dispatch on overloaded functions.")
            if function.type_reflection is not None:
                raise ValueError("Cannot use raw dispatch on type methods.")

            reflection = function.reflections[0]
            slang_function = function.module.layout.find_function(reflection, None)

            # Ensure the function has no out or inout parameters, and no return value.
            if any([x.io_type != IOType.inn for x in slang_function.parameters]):
                raise ValueError(
                    "Raw dispatch functions cannot have out or inout parameters.")
            if slang_function.have_return_value:
                raise ValueError("Raw dispatch functions cannot have return values.")

            program = None

            # Attempt to get entry point and link program, on the assumption that the function is a valid kernel.
            # We only do this if user is not attempting to specify a different thread group size
            if thread_group_size is None:
                try:
                    ep = function.module.device_module.entry_point(
                        reflection.name, type_conformances)  # @IgnoreException
                    opts = SlangLinkOptions()
                    program = session.link_program(
                        [function.module.device_module]+function.module.link, [ep], opts)
                except RuntimeError as e:
                    if not re.match(r'Entry point \"\w+\" not found.*', e.args[0]):
                        raise e
                    else:
                        program = None

            # If not a valid kernel, need to generate one.
            if program is None:

                # Check parameter requirements for raw dispatch kernel gen.
                if len(reflection.parameters) < 1:
                    raise ValueError(
                        "To generate raw dispatch functions, first parameter must be a thread id.")
                if not "uint" in slang_function.parameters[0].type.full_name or slang_function.parameters[0].name != "dispatchThreadID":
                    raise ValueError(
                        f"To generate raw dispatch functions, first parameter must be named dispatchThreadID and uint1/2/3, current param is {slang_function.parameters[0].declaration}")

                # Generate params and arguments.
                params = ",".join([f"{slang_function.parameters[0].declaration}: SV_DispatchThreadID"]+[
                                  "uniform " + x.declaration for x in slang_function.parameters[1:]])
                args = ",".join([x.name for x in slang_function.parameters])

                if thread_group_size is None:
                    thread_group_size = uint3(32, 1, 1)

                # Generate mini-kernel for calling the function.
                code = f"""
import "slangpy";
import "{function.module.device_module.name}";
[shader("compute")]
[numthreads({thread_group_size.x}, {thread_group_size.y}, {thread_group_size.z})]
void {reflection.name}_entrypoint({params}) {{
    {reflection.name}({args});
}}          
"""
                # Write the shader to a file for debugging.
                os.makedirs(".temp", exist_ok=True)
                sanitized = re.sub(r"[<>, ]", "_", function.name)
                fn = f".temp/{function.module.name}_{sanitized}_dispatch.slang"
                with open(fn, "w",) as f:
                    f.write(code)

                # Load and link program
                module = session.load_module_from_source(
                    hashlib.sha256(code.encode()).hexdigest()[0:16], code
                )
                ep = module.entry_point(
                    f"{reflection.name}_entrypoint", type_conformances)
                opts = SlangLinkOptions()
                program = session.link_program(
                    [module, function.module.device_module]+function.module.link, [ep], opts)

            self.kernel = device.create_compute_kernel(program)
            self.device = device

        except Exception as e:
            raise e

    def dispatch(self, thread_count: uint3, vars: dict[str, Any] = {}, command_buffer: CommandBuffer | None = None, **kwargs: dict[str, Any]) -> None:

        # Build 'unpacked' args (that handle IThis)
        unpacked_kwargs = {k: unpack_arg(v) for k, v in kwargs.items()}

        # Use python marshalls to convert provided arguments to dispatch args.
        call_data = {}
        self.runtime.write_raw_dispatch_data(call_data, **unpacked_kwargs)

        # Call dispatch
        self.kernel.dispatch(thread_count, vars, command_buffer, **call_data)

        # If just adding to command buffer, post dispatch is redundant
        if command_buffer is not None:
            return

        # Push updated 'this' values back to original objects
        for (k, arg) in kwargs.items():
            pack_arg(arg, unpacked_kwargs[k])
