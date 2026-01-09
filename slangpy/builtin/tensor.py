# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from __future__ import annotations

from typing import Any, Optional, cast
from slangpy.types.tensor import Tensor
from slangpy.core.native import NativeTensorMarshall, NativeTensor

from slangpy import ShaderObject, ShaderCursor, BufferUsage
from slangpy.reflection import (
    SlangProgramLayout,
    SlangType,
    ArrayType,
    ScalarType,
    MatrixType,
    TensorType,
    TensorAccess,
)
from slangpy.bindings import (
    PYTHON_TYPES,
    BindContext,
    BoundVariable,
    CodeGenBlock,
    ReturnContext,
)
import slangpy.builtin.tensorcommon as spytc


def types_equal(a: SlangType, b: SlangType):
    # TODO: Exact comparison of slang types is not currently possible, and we do the next closest thing
    # of comparing their fully qualified names. This will give false positives on types from different
    # modules but with the same name, and false negatives on the same type with different names
    # (e.g. via typedef)
    return a.full_name == b.full_name


def is_nested_array(a: SlangType):
    while True:
        if isinstance(a, ScalarType):
            return True
        if isinstance(a, MatrixType):
            return True
        if not isinstance(a, ArrayType):
            return False
        if a.element_type is None:
            return False
        a = a.element_type


class TensorMarshall(NativeTensorMarshall):
    def __init__(
        self,
        layout: SlangProgramLayout,
        element_type: SlangType,
        dims: int,
        writable: bool,
        d_in: Optional[TensorMarshall],
        d_out: Optional[TensorMarshall],
    ):

        # Fix up some typings
        self.d_in: Optional[TensorMarshall]
        self.d_out: Optional[TensorMarshall]
        self.slang_type: SlangType
        self.slang_element_type: SlangType
        self.layout = layout

        if d_in is not None or d_out is not None:
            grad_type = element_type.derivative

            if d_in is not None and not types_equal(grad_type, d_in.slang_element_type):
                raise ValueError(
                    f"Invalid element type of input gradient for tensor of type {element_type.full_name}: Expected "
                    f"{grad_type.full_name}, received {d_in.slang_element_type.full_name}"
                )
            if d_out is not None and not types_equal(grad_type, d_out.slang_element_type):
                raise ValueError(
                    f"Invalid element type of output gradient for tensor of type {element_type.full_name}: Expected "
                    f"{grad_type.full_name}, received {d_out.slang_element_type.full_name}"
                )

            if d_in is not None and not writable:
                raise ValueError(
                    "Supplying input gradients is only allowed if the primal tensor is writable"
                )

        slang_type = layout.tensor_type(
            element_type=element_type,
            dims=dims,
            access=TensorAccess.read_write if writable else TensorAccess.read,
            tensor_type=(
                TensorType.difftensor
                if (d_in is not None or d_out is not None)
                else TensorType.tensor
            ),
        )

        if not slang_type:
            raise ValueError(
                f"Failed to find tensor type to contain element {element_type.full_name}. If using differentiable tensors, this can imply\
                             that the element type does not support both the IDifferentiable and IAtomicAddable interfaces."
            )

        super().__init__(
            dims=dims,
            writable=writable,
            slang_type=slang_type,
            slang_element_type=element_type,
            element_layout=element_type.buffer_layout.reflection,
            d_in=d_in,
            d_out=d_out,
        )

    def __repr__(self) -> str:
        return f"Tensor[dtype={self.slang_element_type.full_name}, dims={self.dims}, writable={self.writable}]"

    @property
    def has_derivative(self):
        return self.d_in is not None or self.d_out is not None

    @property
    def is_writable(self):
        return self.writable

    def resolve_types(self, context: BindContext, bound_type: SlangType):
        return spytc.resolve_types(self, context, bound_type)

    def reduce_type(self, context: BindContext, dimensions: int):
        return spytc.reduce_type(self, context, dimensions)

    def resolve_dimensionality(
        self,
        context: BindContext,
        binding: BoundVariable,
        vector_target_type: "SlangType",
    ):
        return spytc.resolve_dimensionality(self, context, binding, vector_target_type)

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: BoundVariable):
        return spytc.gen_calldata(self, cgb, context, binding)

    def build_shader_object(self, context: "BindContext", data: Any) -> "ShaderObject":
        so = context.device.create_shader_object(self.slang_type.uniform_layout.reflection)
        cursor = ShaderCursor(so)
        if not self.has_derivative:
            cursor.write(data.uniforms())
        else:
            cursor["primal"].write(data.uniforms())
            if self.d_in is not None:
                cursor["d_in"].write(data.grad_in.uniforms())
            if self.d_out is not None:
                cursor["d_out"].write(data.grad_out.uniforms())

        cursor.write(data.uniforms())
        return so


def create_tensor_marshall(layout: SlangProgramLayout, value: Any):
    if isinstance(value, NativeTensor):
        d_in = create_tensor_marshall(layout, value.grad_in) if value.grad_in is not None else None
        d_out = (
            create_tensor_marshall(layout, value.grad_out) if value.grad_out is not None else None
        )

        return TensorMarshall(
            layout,
            cast(SlangType, value.dtype),
            len(value.shape),
            (value.usage & BufferUsage.unordered_access) != 0,
            d_in,
            d_out,
        )
    elif isinstance(value, ReturnContext):
        return TensorMarshall(
            layout,
            value.slang_type,
            value.bind_context.call_dimensionality,
            True,
            None,
            None,
        )
    else:
        raise ValueError(f"Type {type(value)} is unsupported for TensorMarshall")


PYTHON_TYPES[NativeTensor] = create_tensor_marshall
PYTHON_TYPES[Tensor] = create_tensor_marshall
