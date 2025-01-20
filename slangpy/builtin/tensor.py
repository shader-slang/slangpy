# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any, Optional

from slangpy.core.native import AccessType, CallContext, CallMode, Shape

from slangpy.reflection.reflectiontypes import is_matching_array_type, VectorType
from slangpy.types.tensor import Tensor, innermost_type

from slangpy.backend import TypeReflection
from slangpy.reflection import TYPE_OVERRIDES, SlangProgramLayout, SlangType, TypeReflection, ArrayType, ScalarType
from slangpy.bindings import (PYTHON_TYPES, Marshall, BindContext,
                              BoundVariable, BoundVariableRuntime,
                              CodeGenBlock, ReturnContext)


class ITensorType(SlangType):
    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):
        args = program.get_resolved_generic_args(refl)
        assert args is not None
        assert len(args) == 2
        assert isinstance(args[0], SlangType)
        assert isinstance(args[1], int)
        super().__init__(program, refl,
                         element_type=args[0], local_shape=Shape((-1,)*args[1]))
        self.element_type: SlangType
        self._writable = refl.name in (
            "IRWTensor", "RWTensor", "GradInTensor", "GradInOutTensor")
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
        if not isinstance(a, ArrayType):
            return False
        if a.element_type is None:
            return False
        a = a.element_type


def build_tensor_name(element_type: SlangType, dims: int, writable: bool, has_grad_in: bool, has_grad_out: bool) -> str:
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


def build_tensor_type(layout: SlangProgramLayout, element_type: SlangType, dims: int, writable: bool, has_grad_in: bool, has_grad_out: bool) -> SlangType:
    tensor_name = build_tensor_name(element_type, dims, writable, has_grad_in, has_grad_out)
    slang_type = layout.find_type_by_name(tensor_name)
    assert slang_type is not None, f"Failed to look up tensor type {tensor_name}"
    return slang_type


class TensorMarshall(Marshall):
    def __init__(self, layout: SlangProgramLayout, element_type: SlangType, dims: int, writable: bool,
                 d_in: Optional[TensorMarshall], d_out: Optional[TensorMarshall]):
        super().__init__(layout)
        self.layout = layout
        self.element_type = element_type
        self.dims = dims
        self.writable = writable
        self.d_in = d_in
        self.d_out = d_out

        self.has_grads = self.d_in is not None or self.d_out is not None

        if not element_type.differentiable:
            raise ValueError(
                f"Tensor element types must be differentiable (received {element_type.full_name})")

        if self.has_grads:
            grad_type = element_type.derivative

            if d_in is not None and not types_equal(grad_type, d_in.element_type):
                raise ValueError(
                    f"Invalid element type of input gradient for tensor of type {element_type.full_name}: Expected "
                    f"{grad_type.full_name}, received {d_in.element_type.full_name}")
            if d_out is not None and not types_equal(grad_type, d_out.element_type):
                raise ValueError(
                    f"Invalid element type of output gradient for tensor of type {element_type.full_name}: Expected "
                    f"{grad_type.full_name}, received {d_out.element_type.full_name}")

            if d_in is not None and not writable:
                raise ValueError(
                    "Supplying input gradients is only allowed if the primal tensor is writable")

        self.slang_type = build_tensor_type(
            layout, element_type, dims, writable, d_in is not None, d_out is not None)

    @property
    def is_writable(self) -> bool:
        return self.writable

    @property
    def has_derivative(self) -> bool:
        return self.has_grads

    def resolve_type(self, context: BindContext, bound_type: SlangType):
        if isinstance(bound_type, ITensorType):
            # Trying to pass to tensor. Verify that we're only narrowing
            if bound_type.writable and not self.writable:
                raise ValueError(
                    "Can't pass a read-only tensor to a writable tensor")
            if not types_equal(bound_type.dtype, self.element_type):
                raise ValueError(
                    f"Can't convert tensor with element type {self.element_type.full_name} "
                    f"to tensor with element type {bound_type.dtype.full_name}")

            return build_tensor_type(self.layout, bound_type.dtype, bound_type.dims, bound_type.writable, self.d_in is not None, self.d_out is not None)

        # if implicit element casts enabled, allow conversion from type to element type
        if context.options['implicit_element_casts']:
            if types_equal(self.element_type, bound_type):
                return bound_type
            if is_matching_array_type(bound_type, self.element_type):
                return self.element_type

        # if implicit tensor casts enabled, allow conversion from vector to element type
        # or array type
        if context.options['implicit_tensor_casts']:
            # Be lenient and allow e.g. passing float to float[N] or float[M][N]
            if types_equal(self.element_type, innermost_type(bound_type)):
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
            return self.element_type
        else:
            raise ValueError("Cannot reduce dimensions of Tensor")

        # TODO: Can't handle case of mapping smaller number of dimensions,
        # because there are multiple possible types we could reduce to
        # (e.g. array vs vector) and we are not allowed to peek at the parameter type

    def resolve_dimensionality(self, context: BindContext, binding: BoundVariable, vector_target_type: 'SlangType'):
        if isinstance(vector_target_type, ITensorType):
            return self.dims - vector_target_type.dims
        else:
            return self.dims + len(self.element_type.shape) - len(vector_target_type.shape)

    def get_shape(self, value: Optional[Tensor] = None) -> Shape:
        if value is not None:
            return Shape(value.shape)
        else:
            return Shape((-1,) * self.dims)

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: BoundVariable):
        if isinstance(binding.vector_type, ITensorType):
            writable = binding.vector_type.writable
        else:
            writable = binding.access[0] in (AccessType.write, AccessType.readwrite)

        type_name = build_tensor_name(
            self.element_type, self.dims, writable, self.d_in is not None, self.d_out is not None)
        cgb.type_alias(f"_t_{binding.variable_name}", type_name)

        # cgb.add_import("tensor")

    def create_calldata(self, context: CallContext, binding: 'BoundVariableRuntime', data: Tensor) -> Any:
        strides = tuple(0 if dim == 1 else stride for dim, stride in zip(data.shape, data.strides))

        primal_calldata = {
            'buffer': data.storage,
            'layout': {'offset': data.offset, 'strides': strides},
            'shape': data.shape
        }

        if not self.d_in and not self.d_out:
            return primal_calldata

        result = {'primal': primal_calldata}
        if self.d_in is not None:
            if data.grad_in is None:
                raise ValueError("Missing required input gradients")
            result['d_in'] = self.d_in.create_calldata(context, binding, data.grad_in)
        if self.d_out is not None:
            if data.grad_out is None:
                raise ValueError("Missing tensor to hold output gradients")
            result['d_out'] = self.d_out.create_calldata(context, binding, data.grad_out)

        if context.call_mode != CallMode.prim and data.grad_in is not None and data.grad_in is data.grad_out:
            if binding.access[1] == AccessType.readwrite:
                raise ValueError(
                    "inout parameter gradients need separate buffers for inputs and outputs (see Tensor.with_grads)")

        return result

    def create_output(self, context: CallContext, binding: BoundVariableRuntime) -> Any:
        return Tensor.empty(context.device, context.call_shape.as_tuple(), self.element_type)

    def read_output(self, context: CallContext, binding: BoundVariableRuntime, data: Any) -> Any:
        return data


def create_tensor_marshall(layout: SlangProgramLayout, value: Any):
    if isinstance(value, Tensor):
        d_in = create_tensor_marshall(
            layout, value.grad_in) if value.grad_in is not None else None
        d_out = create_tensor_marshall(
            layout, value.grad_out) if value.grad_out is not None else None

        return TensorMarshall(layout, value.dtype, len(value.shape), True, d_in, d_out)
    elif isinstance(value, ReturnContext):
        return TensorMarshall(layout, value.slang_type, value.bind_context.call_dimensionality, True, None, None)
    else:
        raise ValueError(f"Type {type(value)} is unsupported for TensorMarshall")


PYTHON_TYPES[Tensor] = create_tensor_marshall
