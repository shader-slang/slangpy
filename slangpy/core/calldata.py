# SPDX-License-Identifier: Apache-2.0
import hashlib
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from slangpy.core.callsignature import *
from slangpy.core.logging import (bound_call_table, bound_exception_info,
                                  mismatch_info)
from slangpy.core.native import CallMode, NativeCallData

from slangpy.backend import SlangCompileError, SlangLinkOptions
from slangpy.bindings import (BindContext, BoundCallRuntime,
                              BoundVariableException, CodeGen)
from slangpy.bindings.boundvariable import BoundCall, BoundVariable
from slangpy.reflection import SlangFunction

if TYPE_CHECKING:
    from slangpy.core.function import FunctionNode

SLANG_PATH = Path(__file__).parent.parent / "slang"


def unpack_arg(arg: Any) -> Any:
    if hasattr(arg, "get_this"):
        arg = arg.get_this()
    if isinstance(arg, dict):
        arg = {k: unpack_arg(v) for k, v in arg.items()}
    if isinstance(arg, list):
        arg = [unpack_arg(v) for v in arg]
    return arg


def pack_arg(arg: Any, unpacked_arg: Any):
    if hasattr(arg, "update_this"):
        arg.update_this(unpacked_arg)
    if isinstance(arg, dict):
        for k, v in arg.items():
            pack_arg(v, unpacked_arg[k])
    if isinstance(arg, list):
        for i, v in enumerate(arg):
            pack_arg(v, unpacked_arg[i])
    return arg


class CallData(NativeCallData):
    def __init__(
        self,
        func: "FunctionNode",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        try:
            # These will be populated later
            bindings = None
            slang_function = None

            # Read temps from function
            function = func
            build_info = function.calc_build_info()
            return_type = build_info.return_type
            positional_mapping = build_info.map_args
            keyword_mapping = build_info.map_kwargs
            type_conformances = build_info.type_conformances

            # Store layout and callmode from function
            self.layout = build_info.module.layout
            self.call_mode = build_info.call_mode

            # Build 'unpacked' args (that handle IThis)
            unpacked_args = tuple([unpack_arg(x) for x in args])
            unpacked_kwargs = {k: unpack_arg(v) for k, v in kwargs.items()}

            # Setup context
            context = BindContext(self.layout, self.call_mode,
                                  build_info.module.device_module, build_info.options)

            # Build the unbound signature from inputs
            bindings = BoundCall(context, *unpacked_args, **unpacked_kwargs)

            # Apply explicit to the Python variables
            apply_explicit_vectorization(
                context, bindings, positional_mapping, keyword_mapping)

            # Perform specialization to get a concrete function reflection
            slang_function = specialize(
                context, bindings, build_info.reflections, build_info.type_reflection)
            if isinstance(slang_function, MismatchReason):
                raise KernelGenException(
                    f"Function signature mismatch: {slang_function.reason}\n\n"
                    f"{mismatch_info(bindings, build_info.reflections)}\n")

            # Check for differentiability error
            if not slang_function.differentiable and self.call_mode != CallMode.prim:
                raise KernelGenException(
                    f"Could not call function '{function.name}': Function is not differentiable\n\n"
                    f"{mismatch_info(bindings, build_info.reflections)}\n")

            # Inject a dummy node into the Python signature if we need a result back
            if self.call_mode == CallMode.prim and not "_result" in kwargs and slang_function.return_type is not None and slang_function.return_type.full_name != 'void':
                rvalnode = BoundVariable(context, None, None, "_result")
                bindings.kwargs["_result"] = rvalnode

           # Create bound variable information now that we have concrete data for path sides
            bindings = bind(context, bindings, slang_function)

            # Run Python side implicit vectorization to do any remaining type resolution
            apply_implicit_vectorization(context, bindings)

            # Should no longer have implicit argument types for anything.
            assert not bindings.has_implicit_args

            # Calculate overall call dimensionality now that all typing is known.
            self.call_dimensionality = calculate_call_dimensionality(bindings)
            context.call_dimensionality = self.call_dimensionality

            # If necessary, create return value node once call dimensionality is known.
            create_return_value_binding(context, bindings, return_type)

            # Calculate final mappings for bindings that only have known vector type.
            finalize_mappings(context, bindings)

            # Should no longer have any unresolved mappings for anything.
            assert not bindings.has_implicit_mappings

            # Validate the arguments we're going to pass to slang before trying to make code.
            validate_specialize(context, bindings, slang_function)

            # Calculate differentiability of all variables.
            calculate_differentiability(context, bindings)

            # Generate code.
            codegen = CodeGen()
            generate_code(context, build_info, bindings, codegen)
            for link in build_info.module.link:
                codegen.add_import(link.name)
            code = codegen.finish(call_data=True, input_load_store=True,
                                  header=True, kernel=True, imports=True,
                                  trampoline=True, context=True, snippets=True,
                                  call_data_structs=True, constants=True)

            # Write the shader to a file for debugging.
            os.makedirs(".temp", exist_ok=True)
            santized_module = re.sub(r"[<>, ./]", "_", build_info.module.name)
            sanitized = re.sub(r"[<>, ./]", "_", build_info.name)
            fn = f".temp/{santized_module}_{sanitized}{'_backwards' if self.call_mode == CallMode.bwds else ''}.slang"

            # with open(fn,"r") as f:
            #    code = f.read()

            with open(fn, "w",) as f:
                f.write("/*\n")
                f.write(bound_call_table(bindings))
                f.write("\n*/\n")
                f.write(code)

            # Hash the code to get a unique identifier for the module.
            # We add type conformances to the start of the code to ensure that the hash is unique
            assert function.slangpy_signature is not None
            code_minus_header = "[CallData]\n" + str(build_info.type_conformances) + \
                code[len(codegen.header):]
            hash = hashlib.sha256(code_minus_header.encode()).hexdigest()

            # Check if we've already built this module.
            if hash in build_info.module.kernel_cache:
                # Get kernel from cache if we have
                self.kernel = build_info.module.kernel_cache[hash]
                self.device = build_info.module.device
            else:
                # Build new module and link it with the one that contains the function being called.
                session = build_info.module.session
                device = session.device
                module = session.load_module_from_source(hash, code)
                ep = module.entry_point(f"main", type_conformances)
                opts = SlangLinkOptions()
                # opts.dump_intermediates = False
                program = session.link_program(
                    [module, build_info.module.device_module]+build_info.module.link, [ep], opts)
                self.kernel = device.create_compute_kernel(program)
                build_info.module.kernel_cache[hash] = self.kernel
                self.device = device

            # Store the bindings and runtime for later use.
            self.debug_only_bindings = bindings
            self.runtime = BoundCallRuntime(bindings)

        except BoundVariableException as e:
            if bindings is not None:
                ref = slang_function.reflection if isinstance(
                    slang_function, SlangFunction) else build_info.reflections[0]
                raise ValueError(
                    f"{e.message}\n\n"
                    f"{bound_exception_info(bindings, ref, e.variable)}\n")
            else:
                raise e
        except SlangCompileError as e:
            if bindings is not None:
                ref = slang_function.reflection if isinstance(
                    slang_function, SlangFunction) else build_info.reflections[0]
                raise ValueError(
                    f"Slang compilation error: {e}\n. See .temp directory for generated shader.\n"
                    f"This most commonly occurs as a result of an invalid explicit type cast, or bug in implicit casting logic.\n"
                    f"{bound_exception_info(bindings, ref, None)}\n")
            else:
                raise e
        except KernelGenException as e:
            if bindings is not None:
                ref = slang_function.reflection if isinstance(
                    slang_function, SlangFunction) else build_info.reflections[0]
                raise ValueError(
                    f"Exception in kernel generation: {e.message}\n."
                    f"{bound_exception_info(bindings, ref, None)}\n")
            else:
                raise e
        except Exception as e:
            if bindings is not None:
                ref = slang_function.reflection if isinstance(
                    slang_function, SlangFunction) else build_info.reflections[0]
                raise ValueError(
                    f"Exception in kernel generation: {e}\n."
                    f"{bound_exception_info(bindings, ref, None)}\n")
            else:
                raise e
