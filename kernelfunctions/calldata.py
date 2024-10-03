import hashlib
import os
import re
from typing import TYPE_CHECKING, Any, Optional

from sgl import CommandBuffer

from kernelfunctions.core import CallMode, PythonFunctionCall, PythonVariable, CodeGen, BoundCallRuntime, NativeCallData

from kernelfunctions.callsignature import (
    bind,
    calculate_call_dimensionality,
    create_return_value_binding,
    finalize_transforms, generate_code,
    generate_tree_info_string,
    get_readable_func_string,
    get_readable_signature_string,
    match_signatures
)

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
            FunctionChainInputTransform,
            FunctionChainOutputTransform,
            FunctionChainSet,
            FunctionChainHook,
            TDispatchHook
        )

        if not isinstance(chain[0], Function):
            raise ValueError("First entry in chain should be a function")
        self.call_mode = CallMode.prim
        self.before_dispatch_hooks: Optional[list[TDispatchHook]] = None
        self.after_dispatch_hooks: Optional[list[TDispatchHook]] = None

        function = chain[0]
        chain = chain
        input_transforms: dict[str, 'TShapeOrTuple'] = {}
        outut_transforms: dict[str, 'TShapeOrTuple'] = {}

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
            if isinstance(item, FunctionChainInputTransform):
                input_transforms.update(item.transforms)
            if isinstance(item, FunctionChainOutputTransform):
                outut_transforms.update(item.transforms)
            if isinstance(item, FunctionChainBwdsDiff):
                self.call_mode = CallMode.bwds
            if isinstance(item, FunctionChainHook):
                if item.before_dispatch is not None:
                    if self.before_dispatch_hooks is None:
                        self.before_dispatch_hooks = []
                    self.before_dispatch_hooks.append(item.before_dispatch)
                if item.after_dispatch is not None:
                    if self.after_dispatch_hooks is None:
                        self.after_dispatch_hooks = []
                    self.after_dispatch_hooks.append(item.after_dispatch)

        self.sets = sets

        # Build 'unpacked' args (that handle IThis)
        unpacked_args = tuple([unpack_arg(x) for x in args])
        unpacked_kwargs = {k: unpack_arg(v) for k, v in kwargs.items()}

        # Build the unbound signature from inputs
        python_call = PythonFunctionCall(*unpacked_args, **unpacked_kwargs)

        # Attempt to match to a slang function overload
        python_to_slang_mapping = None
        slang_function = None
        for overload in function.overloads:
            match = match_signatures(
                python_call, overload, self.call_mode)
            if match:
                if python_to_slang_mapping == None:
                    python_to_slang_mapping = match
                    slang_function = overload
                else:
                    err_text = f"""
Multiple matching overloads found for function {function.name}.
Input signature:
{get_readable_signature_string(python_call)}
First match: {get_readable_func_string(slang_function)}
Second match: {get_readable_func_string(overload)}"""
                    raise ValueError(err_text.strip())

        if python_to_slang_mapping is None or slang_function is None:
            olstrings = "\n".join([get_readable_func_string(x)
                                  for x in function.overloads])
            err_text = f"""
No matching overload found for function {function.name}.
Input signature:
{get_readable_signature_string(python_call)}
Overloads:
{olstrings}
"""
            raise ValueError(err_text.strip())

        # Inject a dummy node into both signatures if we need a result back
        if self.call_mode == CallMode.prim and not "_result" in kwargs and slang_function.return_value is not None:
            rvalnode = PythonVariable(None, None, "_result")
            python_call.kwargs["_result"] = rvalnode
            python_to_slang_mapping[rvalnode] = slang_function.return_value

        # Once matched, build the fully bound signature
        bindings = bind(python_call, python_to_slang_mapping, self.call_mode,
                        input_transforms, outut_transforms)

        # calculate call shaping
        self.call_dimensionality = calculate_call_dimensionality(bindings)

        # if necessary, create return value node
        create_return_value_binding(self.call_dimensionality,
                                    bindings, self.call_mode)

        # once overall dimensionality is known, individual binding transforms can be made concrete
        finalize_transforms(self.call_dimensionality, bindings)

        # generate code
        codegen = CodeGen()
        generate_code(self.call_dimensionality, function,
                      bindings, self.call_mode, codegen)

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

    def call(self, *args: Any, **kwargs: Any):
        return self.exec(None, *args, **kwargs)

    def append_to(self, command_buffer: CommandBuffer, *args: Any, **kwargs: Any):
        return self.exec(command_buffer, *args, **kwargs)
