import hashlib
import os
from typing import TYPE_CHECKING, Any, Optional

from kernelfunctions.core import CallMode, PythonFunctionCall, PythonVariable, CodeGen, BoundCallRuntime

from kernelfunctions.backend import slangpynative

from kernelfunctions.callsignature import (
    bind,
    calculate_call_dimensionality,
    create_return_value_binding,
    finalize_transforms, generate_code,
    generate_tree_info_string,
    get_readable_func_string,
    get_readable_signature_string,
    match_signatures,
)

if TYPE_CHECKING:
    from kernelfunctions.shapes import TConcreteShape
    from kernelfunctions.function import FunctionChainBase

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


class CallData:
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
            FunctionChainThis,
            IThis,
        )

        if not isinstance(chain[0], Function):
            raise ValueError("First entry in chain should be a function")
        self.function = chain[0]
        self.chain = chain
        self.call_mode = CallMode.prim
        self.input_transforms: dict[str, 'TConcreteShape'] = {}
        self.outut_transforms: dict[str, 'TConcreteShape'] = {}
        self.this: Optional[IThis] = None
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
                self.input_transforms.update(item.transforms)
            if isinstance(item, FunctionChainOutputTransform):
                self.outut_transforms.update(item.transforms)
            if isinstance(item, FunctionChainThis):
                self.this = item.this
            if isinstance(item, FunctionChainBwdsDiff):
                self.call_mode = CallMode.bwds

        self.sets = sets

        # If 'this' is specified, inject as first argument
        if self.this is not None:
            args = (self.this,) + args

        # Build 'unpacked' args (that handle IThis)
        unpacked_args = tuple([unpack_arg(x) for x in args])
        unpacked_kwargs = {k: unpack_arg(v) for k, v in kwargs.items()}

        # Build the unbound signature from inputs
        python_call = PythonFunctionCall(*unpacked_args, **unpacked_kwargs)

        # Attempt to match to a slang function overload
        python_to_slang_mapping = None
        slang_function = None
        for overload in self.function.overloads:
            match = match_signatures(
                python_call, overload, self.call_mode)
            if match:
                if python_to_slang_mapping == None:
                    python_to_slang_mapping = match
                    slang_function = overload
                else:
                    err_text = f"""
Multiple matching overloads found for function {self.function.name}.
Input signature:
{get_readable_signature_string(python_call)}
First match: {get_readable_func_string(slang_function)}
Second match: {get_readable_func_string(overload)}"""
                    raise ValueError(err_text.strip())

        if python_to_slang_mapping is None or slang_function is None:
            olstrings = "\n".join([get_readable_func_string(x)
                                  for x in self.function.overloads])
            err_text = f"""
No matching overload found for function {self.function.name}.
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
                        self.input_transforms, self.outut_transforms)

        # store overload and signature
        self.overload = slang_function

        # calculate call shaping
        self.call_dimensionality = calculate_call_dimensionality(bindings)

        # if necessary, create return value node
        create_return_value_binding(self.call_dimensionality,
                                    bindings, self.call_mode)

        # once overall dimensionality is known, individual binding transforms can be made concrete
        finalize_transforms(self.call_dimensionality, bindings)

        # generate code
        codegen = CodeGen()
        generate_code(self.call_dimensionality, self.function,
                      bindings, self.call_mode, codegen)

        # store code
        self.code = codegen.finish(call_data=True, input_load_store=True,
                                   header=True, kernel=True, imports=True,
                                   trampoline=True, context=True, snippets=True,
                                   call_data_structs=True)

        # Write the shader to a file for debugging.
        os.makedirs(".temp", exist_ok=True)
        fn = f".temp/{self.function.module.name}_{self.function.name}{'_backwards' if self.call_mode == CallMode.bwds else ''}.slang"

        # with open(fn,"r") as f:
        #   self.code = f.read()

        with open(fn, "w",) as f:
            f.write("/*\n")
            f.write(generate_tree_info_string(bindings))
            f.write("\n*/\n")
            f.write(self.code)

        # Build new module and link it with the one that contains the function being called.
        session = self.function.module.session
        device = session.device
        module = session.load_module_from_source(
            hashlib.sha256(self.code.encode()).hexdigest()[0:16], self.code
        )
        ep = module.entry_point("main")
        program = session.link_program([module, self.function.module], [ep])
        self.kernel = device.create_compute_kernel(program)

        self.debug_only_bindings = bindings
        self.runtime = BoundCallRuntime(bindings)

    def call(self, *args: Any, **kwargs: Any):

        call_data = {}
        session = self.function.module.session
        device = session.device
        rv_node = None

        # If 'this' is specified, inject as first argument
        if self.this is not None:
            args = (self.this,) + args

        # Build 'unpacked' args (that handle IThis)
        unpacked_args = tuple([unpack_arg(x) for x in args])
        unpacked_kwargs = {k: unpack_arg(v) for k, v in kwargs.items()}

        # Calculate call shape
        self.call_shape = calculate_call_shape(
            self.call_dimensionality, self.runtime, *unpacked_args, **unpacked_kwargs)

        # Allocate a return value if not provided in kw args
        rv_node = self.runtime.kwargs.get("_result", None)
        if self.call_mode == CallMode.prim and rv_node is not None and kwargs.get("_result", None) is None:
            kwargs["_result"] = rv_node._create_output(
                device, self.call_shape)
            unpacked_kwargs["_result"] = kwargs["_result"]

        write_calldata_pre_dispatch(device, self.call_shape, self.runtime,
                                    call_data, *unpacked_args, **unpacked_kwargs)

        # return

        total_threads = 1
        strides = []
        for dim in reversed(self.call_shape):
            strides.append(total_threads)
            total_threads *= dim
        strides.reverse()

        if len(strides) > 0:
            call_data["_call_stride"] = strides
            call_data["_call_dim"] = list(self.call_shape)
        call_data["_thread_count"] = uint3(total_threads, 1, 1)

        # Dispatch the kernel.
        self.kernel.dispatch(uint3(total_threads, 1, 1), {"call_data": call_data})

        read_call_data_post_dispatch(
            device, self.runtime, call_data, *unpacked_args, **unpacked_kwargs)

        # Push updated 'this' values back to original objects
        for (i, arg) in enumerate(args):
            pack_arg(arg, unpacked_args[i])
        for (k, arg) in kwargs.items():
            pack_arg(arg, unpacked_kwargs[k])

        if self.call_mode == CallMode.prim and rv_node is not None:
            return rv_node.read_output(device, kwargs["_result"])
        else:
            return None
