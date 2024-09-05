import os
from typing import Any

from kernelfunctions.callsignature import apply_signature, build_signature, calculate_and_apply_call_shape, generate_code, match_signature
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
from kernelfunctions.signaturenode import CallMode
import kernelfunctions.codegen as cg

TYPES = r"""
int _idx<let N: int>(int[N] index, int[N] stride) {
    int idx = 0;
    for (int i = 0; i < N; i++) {
        idx += index[i] * stride[i];
    }
    return idx;
}

struct TensorBuffer<T, let N : int> {
    RWStructuredBuffer<T> buffer;
    int[N] strides;
    T get(int[N] index) {
        return buffer[_idx(index, strides)];
    }
    __subscript(int[N] index)->T
    {
        get { return get(index); }
    }
}

struct RWTensorBuffer<T, let N : int> {
    RWStructuredBuffer<T> buffer;
    int[N] strides;
    T get(int[N] index) {
        return buffer[_idx(index, strides)];
    }
    void set(int[N] index, T value) {
        buffer[_idx(index, strides)] = value;
    }
    __subscript(int[N] index)->T
    {
        get { return get(index); }
        set { set(index, newValue); }
    }
}
"""


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
        self.args = args
        self.kwargs = kwargs
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
        unbound_signature = build_signature(*self.args, **self.kwargs)

        # Attempt to match
        matched_signature = None
        matched_overload = None
        for ast_function in self.function.ast_functions:
            match = match_signature(
                unbound_signature, ast_function.as_function(), self.call_mode)
            if match:
                if matched_signature == None:
                    matched_signature = match
                    matched_overload = ast_function.as_function()
                else:
                    raise ValueError("Amiguous call - multiple matching overloads found")
        if matched_signature is None or matched_overload is None:
            raise ValueError("No matching overload found")

        # Once matched, build the fully bound signature
        apply_signature(matched_signature, matched_overload,
                        self.input_transforms, self.outut_transforms)

        # store overload and signature
        self.signature = matched_signature
        self.overload = matched_overload

        # calculate call shaping
        self.call_shape = calculate_and_apply_call_shape(self.signature)

        # generate code
        codegen = cg.CodeGen()
        codegen.header = TYPES
        generate_code(self.call_shape, self.function,
                      self.signature, self.call_mode, codegen)

        # store code
        self.code = codegen.finish(call_data=True, input_load_store=True,
                                   header=True, kernel=True, imports=True, trampoline=True)

        # Write the shader to a file for debugging.
        os.makedirs(".temp", exist_ok=True)
        with open(
            f".temp/{self.function.module.name}_{self.function.name}{'_backwards' if self.call_mode == CallMode.bwds else ''}.slang",
            "w",
        ) as f:
            f.write(self.code)
