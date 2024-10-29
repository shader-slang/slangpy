import hashlib
import os
import re
from typing import TYPE_CHECKING, Any

from kernelfunctions.core import CallMode, PythonFunctionCall, PythonVariable, CodeGen, BindContext, BoundCallRuntime, NativeCallData

from kernelfunctions.callsignature import (
    apply_bindings,
    apply_explicit_vectorization,
    apply_implicit_vectorization,
    bind,
    calculate_call_dimensionality,
    create_return_value_binding,
    finalize_mappings,
    finalize_transforms, generate_code,
    generate_tree_info_string,
    get_readable_func_string,
    get_readable_signature_string,
    MismatchReason,
    specialize
)
from kernelfunctions.core.slangvariable import SlangFunction

if TYPE_CHECKING:
    from kernelfunctions.function import FunctionChainBase
    from kernelfunctions.shapes import TShapeOrTuple

SLANG_PATH = os.path.join(os.path.dirname(__file__), "slang")


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
        chain: list["FunctionChainBase"],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        from kernelfunctions.function import (
            Function,
            FunctionChainBwdsDiff,
            FunctionChainOutputTransform,
            FunctionChainSet,
            FunctionChainHook,
            FunctionChainReturnType,
            FunctionChainMap
        )

        if not isinstance(chain[0], Function):
            raise ValueError("First entry in chain should be a function")
        self.call_mode = CallMode.prim

        function = chain[0]
        chain = chain
        outut_transforms: dict[str, 'TShapeOrTuple'] = {}
        return_type = None
        positional_mapping = ()
        keyword_mapping = {}

        sets = {}
        for item in chain:
            if isinstance(item, FunctionChainSet):
                if item.props is not None:
                    sets.update(item.props)
                elif item.callback is not None:
                    sets.update(item.callback(self))
                else:
                    raise ValueError(
                        "FunctionChainSet must have either a props or callback"
                    )
            if isinstance(item, FunctionChainOutputTransform):
                outut_transforms.update(item.transforms)
            if isinstance(item, FunctionChainBwdsDiff):
                self.call_mode = CallMode.bwds
            if isinstance(item, FunctionChainHook):
                if item.before_dispatch is not None:
                    self.add_before_dispatch_hook(item.before_dispatch)
                if item.after_dispatch is not None:
                    self.add_after_dispatch_hook(item.after_dispatch)
            if isinstance(item, FunctionChainReturnType):
                return_type = item.return_type
            if isinstance(item, FunctionChainMap):
                positional_mapping = item.args
                keyword_mapping = item.kwargs

        self.vars = sets

        # Build 'unpacked' args (that handle IThis)
        unpacked_args = tuple([unpack_arg(x) for x in args])
        unpacked_kwargs = {k: unpack_arg(v) for k, v in kwargs.items()}

        # Setup context
        context = BindContext(self.call_mode, function.module)

        # Build the unbound signature from inputs
        python_call = PythonFunctionCall(*unpacked_args, **unpacked_kwargs)

        # Apply explicit to the Python variables
        apply_explicit_vectorization(python_call, positional_mapping, keyword_mapping)

        # Perform specialization to get a concrete function reflection
        concrete_reflection = specialize(context, python_call, function.reflection)
        if isinstance(concrete_reflection, MismatchReason):
            raise ValueError(
                f"Function signature mismatch: {concrete_reflection.reason}\n"
                f"  {get_readable_func_string(function.overloads[0])}\n"
                f"  {get_readable_signature_string(python_call)}")

        # Build slang function signature
        slang_function = SlangFunction(concrete_reflection)

        # Inject a dummy node into the Python signature if we need a result back
        if self.call_mode == CallMode.prim and not "_result" in kwargs and slang_function.return_value is not None:
            rvalnode = PythonVariable(None, None, "_result")
            python_call.kwargs["_result"] = rvalnode

        # Create bound variable information now that we have concrete data for path sides
        bindings = bind(context, python_call, slang_function)

        # Run Python side implicit vectorization to do any remaining type resolution
        apply_implicit_vectorization(context, bindings)

        # Should no longer have implicit argument types for anything.
        assert not python_call.has_implicit_args

        # apply bindings now both python and slang types are finalized
        bindings = apply_bindings(context, bindings)

        # calculate call shaping
        self.call_dimensionality = calculate_call_dimensionality(bindings)
        context.call_dimensionality = self.call_dimensionality

        # Calculate final mappings for bindings that only have known vector type
        finalize_mappings(context, bindings)

        # Should no longer have any unresolved mappings for anything
        assert not python_call.has_implicit_mappings

        # if necessary, create return value node
        create_return_value_binding(context, bindings, return_type)

        # once overall dimensionality is known, individual binding transforms can be made concrete
        finalize_transforms(context, bindings)

        # generate code
        codegen = CodeGen()
        generate_code(context, function, bindings, codegen)

        # store code
        code = codegen.finish(call_data=True, input_load_store=True,
                              header=True, kernel=True, imports=True,
                              trampoline=True, context=True, snippets=True,
                              call_data_structs=True)

        # Write the shader to a file for debugging.
        os.makedirs(".temp", exist_ok=True)
        sanitized = re.sub(r"[<>, ]", "_", function.name)
        fn = f".temp/{function.module.name}_{sanitized}{'_backwards' if self.call_mode == CallMode.bwds else ''}.slang"

        # with open(fn,"r") as f:
        #   self.code = f.read()

        with open(fn, "w",) as f:
            f.write("/*\n")
            f.write(generate_tree_info_string(bindings))
            f.write("\n*/\n")
            f.write(code)

        # Build new module and link it with the one that contains the function being called.
        session = function.module.session
        device = session.device
        module = session.load_module_from_source(
            hashlib.sha256(code.encode()).hexdigest()[0:16], code
        )
        ep = module.entry_point("main")
        program = session.link_program([module, function.module], [ep])
        self.kernel = device.create_compute_kernel(program)
        self.device = device

        self.debug_only_bindings = bindings
        self.runtime = BoundCallRuntime(bindings)
