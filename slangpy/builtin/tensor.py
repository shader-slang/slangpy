# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from __future__ import annotations

from typing import Any, Optional, cast

from slangpy.core.native import AccessType, Shape

from slangpy.reflection.reflectiontypes import is_matching_array_type, VectorType
from slangpy.types.tensor import Tensor
from slangpy.types.buffer import innermost_type
from slangpy.core.native import NativeTensorMarshall, NativeTensor

from slangpy import TypeReflection, ShaderObject, ShaderCursor
from slangpy.reflection import (
    TYPE_OVERRIDES,
    SlangProgramLayout,
    SlangType,
    TypeReflection,
    ArrayType,
    ScalarType,
    MatrixType,
)
from slangpy.bindings import (
    PYTHON_TYPES,
    BindContext,
    BoundVariable,
    CodeGenBlock,
    ReturnContext,
)


class ITensorType(SlangType):
    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):
        args = program.get_resolved_generic_args(refl)
        assert args is not None
        assert len(args) == 2
        assert isinstance(args[0], SlangType)
        assert isinstance(args[1], int)
        super().__init__(program, refl, element_type=args[0], local_shape=Shape((-1,) * args[1]))
        self.element_type: SlangType
        self._writable = refl.name in (
            "IRWTensor",
            "RWTensor",
            "GradInTensor",
            "GradInOutTensor",
            "AtomicTensor",
        )
        self._dims = args[1]

    @property
    def writable(self) -> bool:
        return self._writable

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def dtype(self) -> SlangType:
        return self.element_type


TYPE_OVERRIDES["ITensor"] = ITensorType
TYPE_OVERRIDES["IRWTensor"] = ITensorType
TYPE_OVERRIDES["Tensor"] = ITensorType
TYPE_OVERRIDES["RWTensor"] = ITensorType
TYPE_OVERRIDES["GradInTensor"] = ITensorType
TYPE_OVERRIDES["GradOutTensor"] = ITensorType
TYPE_OVERRIDES["GradInOutTensor"] = ITensorType
TYPE_OVERRIDES["AtomicTensor"] = ITensorType


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


def build_tensor_name(
    element_type: SlangType,
    dims: int,
    writable: bool,
    has_grad_in: bool,
    has_grad_out: bool,
) -> str:
    if not has_grad_in and not has_grad_out:
        prefix = "RW" if writable else ""
    else:
        prefix = "Grad"
        if has_grad_in:
            # assert writable
            prefix += "In"
        if has_grad_out:
            prefix += "Out"
    return f"{prefix}Tensor<{element_type.full_name}, {dims}>"


def build_tensor_type(
    layout: SlangProgramLayout,
    element_type: SlangType,
    dims: int,
    writable: bool,
    has_grad_in: bool,
    has_grad_out: bool,
) -> SlangType:
    tensor_name = build_tensor_name(element_type, dims, writable, has_grad_in, has_grad_out)
    slang_type = layout.find_type_by_name(tensor_name)
    assert slang_type is not None, f"Failed to look up tensor type {tensor_name}"
    return slang_type


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
        self.slang_element_type: SlangType

        self.layout = layout

        if not element_type.differentiable:
            raise ValueError(
                f"Tensor element types must be differentiable (received {element_type.full_name})"
            )

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

        slang_type = build_tensor_type(
            layout, element_type, dims, writable, d_in is not None, d_out is not None
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

    @property
    def has_derivative(self):
        return self.d_in is not None or self.d_out is not None

    @property
    def is_writable(self):
        return self.writable

    def resolve_type(self, context: BindContext, bound_type: SlangType):
        if isinstance(bound_type, ITensorType):
            # Trying to pass to tensor. Verify that we're only narrowing
            if bound_type.writable and not self.writable:
                raise ValueError("Can't pass a read-only tensor to a writable tensor")
            if not types_equal(bound_type.dtype, self.slang_element_type):
                raise ValueError(
                    f"Can't convert tensor with element type {self.slang_element_type.full_name} "
                    f"to tensor with element type {bound_type.dtype.full_name}"
                )

            # Atomic tensors are special, they must be passed as-is
            if bound_type.name == "AtomicTensor":
                return bound_type

            return build_tensor_type(
                self.layout,
                bound_type.dtype,
                bound_type.dims,
                bound_type.writable,
                self.d_in is not None,
                self.d_out is not None,
            )

        # if implicit element casts enabled, allow conversion from type to element type
        if context.options["implicit_element_casts"]:
            if types_equal(self.slang_element_type, bound_type):
                return bound_type
            if is_matching_array_type(bound_type, self.slang_element_type):
                return self.slang_element_type

        # if implicit tensor casts enabled, allow conversion from vector to element type
        # or array type
        if context.options["implicit_tensor_casts"]:
            # Be lenient and allow e.g. passing float to float[N] or float[M][N]
            if types_equal(self.slang_element_type, innermost_type(bound_type)):
                if is_nested_array(bound_type) and len(bound_type.shape) <= 2:
                    return bound_type
                if isinstance(bound_type, VectorType):
                    return bound_type

        # Default to casting to itself
        return self.slang_type

    def reduce_type(self, context: BindContext, dimensions: int):
        if dimensions == 0:
            return self.slang_type
        elif dimensions == self.dims:
            return self.slang_element_type
        else:
            raise ValueError("Cannot reduce dimensions of Tensor")

        # TODO: Can't handle case of mapping smaller number of dimensions,
        # because there are multiple possible types we could reduce to
        # (e.g. array vs vector) and we are not allowed to peek at the parameter type

    def resolve_dimensionality(
        self,
        context: BindContext,
        binding: BoundVariable,
        vector_target_type: "SlangType",
    ):
        if isinstance(vector_target_type, ITensorType):
            return self.dims - vector_target_type.dims
        else:
            return self.dims + len(self.slang_element_type.shape) - len(vector_target_type.shape)

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: BoundVariable):
        if isinstance(binding.vector_type, ITensorType):
            writable = binding.vector_type.writable
        else:
            writable = binding.access[0] in (AccessType.write, AccessType.readwrite)

        # Atomic tensors are special, they must be passed as-is
        if binding.vector_type.name == "AtomicTensor":
            type_name = binding.vector_type.full_name
        else:
            type_name = build_tensor_name(
                self.slang_element_type,
                self.dims,
                writable,
                self.d_in is not None,
                self.d_out is not None,
            )
        cgb.type_alias(f"_t_{binding.variable_name}", type_name)

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
            layout, cast(SlangType, value.dtype), len(value.shape), True, d_in, d_out
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
