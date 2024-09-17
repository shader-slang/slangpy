import hashlib
import os
from typing import Any

from kernelfunctions.core import CallMode, PythonFunctionCall, PythonVariable, CodeGen

from kernelfunctions.backend import uint3

from kernelfunctions.callsignature import (
    bind,
    calculate_and_apply_call_shape,
    create_return_value, generate_code,
    get_readable_func_string,
    get_readable_signature_string,
    match_signatures,
    read_call_data_post_dispatch,
    write_calldata_pre_dispatch
)

from kernelfunctions.function import (
    Function,
    FunctionChainBase,
    FunctionChainInputTransform,
    FunctionChainOutputTransform,
    FunctionChainSet,
)

from kernelfunctions.shapes import (
    TConcreteShape,
)


SLANG_PATH = os.path.join(os.path.dirname(__file__), "slang")


class CallData:
    def __init__(
        self,
        chain: list["FunctionChainBase"],
        backwards: bool,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if not isinstance(chain[0], Function):
            raise ValueError("First entry in chain should be a function")
        self.function = chain[0]
        self.chain = chain
        self.call_mode = CallMode.bwds if backwards else CallMode.prim
        self.input_transforms: dict[str, TConcreteShape] = {}
        self.outut_transforms: dict[str, TConcreteShape] = {}
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

        self.sets = sets

        # Build the unbound signature from inputs
        python_call = PythonFunctionCall(*args, **kwargs)

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
        self.bindings = bind(python_call, python_to_slang_mapping, self.call_mode,
                             self.input_transforms, self.outut_transforms)

        # store overload and signature
        self.overload = slang_function

        # calculate call shaping
        self.call_shape = calculate_and_apply_call_shape(self.bindings)

        # if necessary, create return value node
        create_return_value(self.call_shape, self.bindings, self.call_mode)

        # generate code
        codegen = CodeGen()
        generate_code(self.call_shape, self.function,
                      self.bindings, self.call_mode, codegen)

        # store code
        self.code = codegen.finish(call_data=True, input_load_store=True,
                                   header=True, kernel=True, imports=True,
                                   trampoline=True, context=True, snippets=True,
                                   call_data_structs=True)

        # Write the shader to a file for debugging.
        os.makedirs(".temp", exist_ok=True)
        fn = f".temp/{self.function.module.name}_{self.function.name}{'_backwards' if self.call_mode == CallMode.bwds else ''}.slang"

        with open(fn, "w",) as f:
            f.write(self.code)

        # with open(fn,"r") as f:
        #    self.code = f.read()

        # Build new module and link it with the one that contains the function being called.
        session = self.function.module.session
        device = session.device
        module = session.load_module_from_source(
            hashlib.sha256(self.code.encode()).hexdigest()[0:16], self.code
        )
        ep = module.entry_point("main")
        program = session.link_program([module, self.function.module], [ep])
        self.kernel = device.create_compute_kernel(program)

    def call(self, *args: Any, **kwargs: Any):
        call_data = {}
        session = self.function.module.session
        device = session.device

        # Allocate a return value if not provided in kw args
        rv_node = self.bindings.kwargs.get("_result", None)
        if self.call_mode == CallMode.prim and rv_node is not None and not "_result" in kwargs:
            kwargs["_result"] = rv_node.python.create_output(
                device, self.call_shape)

        write_calldata_pre_dispatch(device, self.bindings,
                                    call_data, *args, **kwargs)

        total_threads = 1
        strides = []
        for dim in reversed(self.call_shape):
            strides.append(total_threads)
            total_threads *= dim
        strides.reverse()

        if len(strides) > 0:
            call_data["_call_stride"] = strides
            call_data["_call_dim"] = self.call_shape
        call_data["_thread_count"] = uint3(total_threads, 1, 1)

        # Dispatch the kernel.
        self.kernel.dispatch(uint3(total_threads, 1, 1), {"call_data": call_data})

        read_call_data_post_dispatch(
            device, self.bindings, call_data, *args, **kwargs)

        if self.call_mode == CallMode.prim and rv_node is not None:
            return rv_node.read_output(device, kwargs["_result"])
        else:
            return None
