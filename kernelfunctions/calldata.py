import hashlib
import os
from typing import Any

from kernelfunctions.backend import uint3

from kernelfunctions.callsignature import apply_signature, build_signature, calculate_and_apply_call_shape, create_return_value, generate_code, get_readable_func_string, get_readable_signature_string, match_signature, read_call_data_post_dispatch, write_calldata_pre_dispatch
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

import kernelfunctions.codegen as cg
from kernelfunctions.types import CallMode
from kernelfunctions.types.pythonvalue import PythonValue


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
        self.input_signature = build_signature(*args, **kwargs)

        # Attempt to match
        python_to_slang_mapping = None
        matched_overload = None
        for overload in self.function.overloads:
            match = match_signature(
                self.input_signature, overload, self.call_mode)
            if match:
                if python_to_slang_mapping == None:
                    python_to_slang_mapping = match
                    matched_overload = overload
                else:
                    err_text = f"""
Multiple matching overloads found for function {self.function.name}.
Input signature:
{get_readable_signature_string(self.input_signature)}
First match: {get_readable_func_string(matched_overload)}
Second match: {get_readable_func_string(overload)}"""
                    raise ValueError(err_text.strip())

        if python_to_slang_mapping is None or matched_overload is None:
            olstrings = "\n".join([get_readable_func_string(x)
                                  for x in self.function.overloads])
            err_text = f"""
No matching overload found for function {self.function.name}.
Input signature:
{get_readable_signature_string(self.input_signature)}
Overloads:
{olstrings}
"""
            raise ValueError(err_text.strip())

        # Inject a dummy node into both signatures if we need a result back
        if self.call_mode == CallMode.prim and not "_result" in kwargs and matched_overload.return_value is not None:
            rvalnode = PythonValue(None, None, "_result")
            self.input_signature.kwargs["_result"] = rvalnode
            python_to_slang_mapping[rvalnode] = matched_overload.return_value

        # Once matched, build the fully bound signature
        self.signature = apply_signature(self.input_signature, python_to_slang_mapping, self.call_mode,
                                         self.input_transforms, self.outut_transforms)

        # store overload and signature
        self.overload = matched_overload

        # calculate call shaping
        self.call_shape = calculate_and_apply_call_shape(self.signature)

        # if necessary, create return value node
        create_return_value(self.call_shape, self.signature, self.call_mode)

        # generate code
        codegen = cg.CodeGen()
        generate_code(self.call_shape, self.function,
                      self.signature, self.call_mode, codegen)

        # store code
        self.code = codegen.finish(call_data=True, input_load_store=True,
                                   header=True, kernel=True, imports=True,
                                   trampoline=True, context=True, snippets=True)

        # Write the shader to a file for debugging.
        os.makedirs(".temp", exist_ok=True)
        with open(
            f".temp/{self.function.module.name}_{self.function.name}{'_backwards' if self.call_mode == CallMode.bwds else ''}.slang",
            "w",
        ) as f:
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

    def call(self, *args: Any, **kwargs: Any):
        call_data = {}
        session = self.function.module.session
        device = session.device

        # Allocate a return value if not provided in kw args
        rv_node = self.signature.kwargs.get("_result", None)
        if self.call_mode == CallMode.prim and rv_node is not None and not "_result" in kwargs:
            kwargs["_result"] = rv_node.python.allocate_return_value(
                device, self.call_shape)

        write_calldata_pre_dispatch(device, self.signature,
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
            device, self.signature, call_data, *args, **kwargs)

        if self.call_mode == CallMode.prim and rv_node is not None:
            return rv_node.python.as_return_value(kwargs["_result"])
        else:
            return None
