# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import hashlib
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from slangpy.core.callsignature import *
from slangpy.core.logging import bound_call_table, bound_exception_info, mismatch_info
from slangpy.core.native import (
    CallMode,
    CallDataMode,
    NativeCallData,
    unpack_refs_and_args,
    unpack_refs_and_kwargs,
    NativeCallRuntimeOptions,
    TensorRef,
)

from slangpy import SlangCompileError, SlangLinkOptions, NativeHandle, DeviceType
from slangpy.bindings import (
    BindContext,
    BoundCallRuntime,
    BoundVariableException,
    CodeGen,
)
from slangpy.bindings.boundvariable import BoundCall, BoundVariable
from slangpy.reflection import SlangFunction

if TYPE_CHECKING:
    from slangpy.core.function import FunctionNode

SLANG_PATH = Path(__file__).parent.parent / "slang"

_DUMP_GENERATED_SHADERS = os.environ.get("SLANGPY_DUMP_GENERATED_SHADERS", "false").lower() in (
    "true",
    "1",
)

_DUMP_SLANG_INTERMEDIATES = os.environ.get("SLANGPY_DUMP_SLANG_INTERMEDIATES", "false").lower() in (
    "true",
    "1",
)
_PRINT_GENERATED_SHADERS = os.environ.get("SLANGPY_PRINT_GENERATED_SHADERS", "false").lower() in (
    "true",
    "1",
)


def set_dump_generated_shaders(value: bool):
    """
    Specify whether to dump generated shaders to .temp for analysis.
    """
    global _DUMP_GENERATED_SHADERS
    _DUMP_GENERATED_SHADERS = value


def set_dump_slang_intermediates(value: bool):
    """
    Specify whether to dump slang compiler intermediates for analysis.
    """
    global _DUMP_SLANG_INTERMEDIATES
    _DUMP_SLANG_INTERMEDIATES = value


def set_print_generated_shaders(value: bool):
    """
    Specify whether to print generated shaders to the terminal for analysis.
    Can also be controlled via the SLANGPY_PRINT_GENERATED_SHADERS environment variable.
    """
    global _PRINT_GENERATED_SHADERS
    _PRINT_GENERATED_SHADERS = value


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
            self.call_group_shape = build_info.call_group_shape

            # Store logger from either function or module
            if build_info.logger is not None:
                self.logger = build_info.logger
            else:
                self.logger = build_info.module.logger
            self.debug_name = f"{build_info.module.name}::{function.name}"

            self.log_debug(f"Generating kernel for {func.name}")
            self.log_debug(f"  Module: {build_info.module.name}")

            # Store layout and callmode from function
            self.layout = build_info.module.layout
            self.call_mode = build_info.call_mode

            # Set call data mode based on device type
            if build_info.module.device.info.type == DeviceType.cuda:
                self.call_data_mode = CallDataMode.entry_point
            else:
                self.call_data_mode = CallDataMode.global_data

            # Build 'unpacked' args (that handle IThis) and extract any pytorch
            # tensor references at the same time.
            tensor_refs = []
            unpacked_args = unpack_refs_and_args(tensor_refs, *args)
            unpacked_kwargs = unpack_refs_and_kwargs(tensor_refs, **kwargs)

            # If we have tensor references, store whether a call to this function
            # will require injection into the autograd graph by checking if any of
            # the tensors require gradients.
            if len(tensor_refs) > 0:
                import torch
                import slangpy.torchintegration.torchtensormarshall  # type: ignore (Causes tensor ref handler to be created)

                autograd = False
                for ref in tensor_refs:
                    assert isinstance(ref, TensorRef), f"Expected TensorRef, got {type(ref)}"
                    assert isinstance(
                        ref.tensor, torch.Tensor
                    ), f"Expected torch.Tensor, got {type(ref.tensor)}"
                    autograd = autograd or ref.tensor.requires_grad
                self.torch_integration = True
                self.torch_autograd = autograd
                if return_type is None:
                    return_type = TensorRef

            # Setup context
            context = BindContext(
                self.layout,
                self.call_mode,
                build_info.module.device_module,
                build_info.options,
                self.call_data_mode,
            )

            # Build the unbound signature from inputs
            bindings = BoundCall(context, *unpacked_args, **unpacked_kwargs)

            # Apply explicit to the Python variables
            apply_explicit_vectorization(context, bindings, positional_mapping, keyword_mapping)

            # Perform specialization to get a concrete function reflection
            slang_function = specialize(
                context, bindings, build_info.function, build_info.this_type
            )
            if isinstance(slang_function, MismatchReason):
                raise ResolveException(
                    f"Function signature mismatch: {slang_function.reason}\n\n"
                    f"{mismatch_info(bindings, build_info.function)}\n"
                )

            # Check for differentiability error
            if not slang_function.differentiable and self.call_mode != CallMode.prim:
                raise ResolveException(
                    f"Could not call function '{function.name}': Function is not differentiable\n\n"
                    f"{mismatch_info(bindings, build_info.function)}\n"
                )

            # Inject a dummy node into the Python signature if we need a result back
            if (
                self.call_mode == CallMode.prim
                and not "_result" in kwargs
                and slang_function.return_type is not None
                and slang_function.return_type.full_name != "void"
            ):
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
            self.log_debug(f"  Call dimensionality: {self.call_dimensionality}")

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
                use_param_block_for_call_data=context.call_data_mode == CallDataMode.global_data,
            )

            # Optionally write the shader to a file for debugging.
            sanitized = ""
            if _DUMP_GENERATED_SHADERS or _DUMP_SLANG_INTERMEDIATES:
                os.makedirs(".temp", exist_ok=True)
                santized_module = re.sub(r"[<>, ./:]", "_", build_info.module.name)
                sanitized = re.sub(r"[:<>, ./:]", "_", build_info.name)
                santized_module = santized_module[:50]
                sanitized = sanitized[:50]
                fn = f".temp/{santized_module}_{sanitized}{'_backwards' if self.call_mode == CallMode.bwds else ''}"
                # Some platforms have path length limits that are easily exceeded with nested generics
                # Be a good citizen here and limit the length of what we generate
                length_limit = 200
                if len(fn) > length_limit:
                    fn = fn[:length_limit]
                fn += "-" + hashlib.sha256(code.encode()).hexdigest()[0:8]
                fn = fn + ".slang"

                # with open(fn,"r") as f:
                #    code = f.read()
                with open(
                    fn,
                    "w",
                ) as f:
                    f.write("/*\n")
                    f.write(bound_call_table(bindings))
                    f.write("\n*/\n")
                    f.write(code)

            # Optionally print the shader to the terminal for AI analysis.
            if _PRINT_GENERATED_SHADERS:
                print("=" * 80)
                print(f"GENERATED SHADER: {build_info.module.name}::{build_info.name}")
                if self.call_mode == CallMode.bwds:
                    print("MODE: Backwards")
                else:
                    print("MODE: Forward")
                print("=" * 80)
                print("/* BINDINGS:")
                print(bound_call_table(bindings))
                print("*/")
                print(code)
                print("=" * 80)
                print(f"END SHADER: {build_info.module.name}::{build_info.name}")
                print("=" * 80)
                print()

            # Hash the code to get a unique identifier for the module.
            # We add type conformances to the start of the code to ensure that the hash is unique
            assert function.slangpy_signature is not None
            code_minus_header = (
                "[CallData]\n" + str(build_info.type_conformances) + code[len(codegen.header) :]
            )
            hash = hashlib.sha256(code_minus_header.encode()).hexdigest()

            # Check if we've already built this module.
            if hash in build_info.module.compute_pipeline_cache:
                # Get pipeline from cache if we have
                self.compute_pipeline = build_info.module.compute_pipeline_cache[hash]
                self.device = build_info.module.device
                self.log_debug(f"  Found cached pipeline with hash {hash}")

            else:
                # Build new module and link it with the one that contains the function being called.
                self.log_debug(f"  Building new pipeline with hash {hash}")
                session = build_info.module.session
                device = session.device
                module = session.load_module_from_source(hash, code)
                ep = module.entry_point(f"compute_main", type_conformances)
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
                build_info.module.compute_pipeline_cache[hash] = self.compute_pipeline
                self.device = device
                self.log_debug(f"  Build succesful")

            # Store the bindings and runtime for later use.
            self.debug_only_bindings = bindings
            self.runtime = BoundCallRuntime(bindings)

        except BoundVariableException as e:
            if bindings is not None:
                ref = (
                    slang_function
                    if isinstance(slang_function, SlangFunction)
                    else build_info.function
                )
                raise ValueError(
                    f"{e.message}\n\n" f"{bound_exception_info(bindings, ref, e.variable)}\n"
                ) from e
            else:
                raise
        except SlangCompileError as e:
            if bindings is not None:
                ref = (
                    slang_function
                    if isinstance(slang_function, SlangFunction)
                    else build_info.function
                )
                raise ValueError(
                    f"Slang compilation error: {e}\n. Use set_dump_generated_shaders to enable dump generated shader to .temp.\n"
                    f"This most commonly occurs as a result of an invalid explicit type cast, or bug in implicit casting logic.\n\n"
                    f"{bound_exception_info(bindings, ref, None)}\n"
                ) from e
            else:
                raise
        except KernelGenException as e:
            if bindings is not None:
                ref = (
                    slang_function
                    if isinstance(slang_function, SlangFunction)
                    else build_info.function
                )
                raise ValueError(
                    f"Exception in kernel generation: {e.message}.\n\n"
                    f"{bound_exception_info(bindings, ref, None)}\n"
                ) from e
            else:
                raise
        except ResolveException as e:
            # Triggered from within calldata, doesn't need augmenting
            raise
        except Exception as e:
            if bindings is not None:
                ref = (
                    slang_function
                    if isinstance(slang_function, SlangFunction)
                    else build_info.function
                )
                raise ValueError(
                    f"Exception in kernel generation: {e}.\n"
                    f"{bound_exception_info(bindings, ref, None)}\n"
                ) from e
            else:
                raise

    def _py_torch_call(
        self,
        function: "FunctionNode",
        options: NativeCallRuntimeOptions,
        args: tuple[Any],
        kwargs: dict[str, Any],
    ) -> Any:
        """
        Call the kernel with the given arguments and options.
        """
        import torch
        from slangpy.torchintegration.autogradhook import TorchAutoGradHook

        # Unpack args and kwargs
        refs: list[TensorRef] = []
        unpacked_args = unpack_refs_and_args(refs, *args)
        unpacked_kwargs = unpack_refs_and_kwargs(refs, **kwargs)

        # Set the cuda stream to use (CUDA backend) or sync to (Vulkan/Metal/D3D12 backend) for the call
        options.cuda_stream = NativeHandle.from_cuda_stream(torch.cuda.current_stream().cuda_stream)

        # Call the kernel
        res = self.call(options, *unpacked_args, **unpacked_kwargs)

        # If result is a tensor ref, ensure its in refs list and extract the torch tensor to return
        if isinstance(res, TensorRef):
            if not res in refs:
                refs.append(res)
            res = cast(torch.Tensor, res.tensor)

        if self.torch_autograd:
            # Extract all tensors that should be treated as inputs to the auto-grad function
            # i.e. ones that SlangPy marked as 'read' or 'readwrite' during the primal call.
            # These can then be passed as arguments to the auto-grad function so they get hooked
            # into the torch auto-grad graph.
            primal_in_tensors = [
                x.tensor
                for x in refs
                if x.last_access[0] in (AccessType.read, AccessType.readwrite)
            ]

            # Call the dummy auto-grad apply function, which critically takes the primal input list
            # as arguments and returns the primal output list as results
            TorchAutoGradHook.apply(
                function,
                unpacked_args,
                unpacked_kwargs,
                refs,
                *primal_in_tensors,
            )

        return res
