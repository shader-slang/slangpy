# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from __future__ import annotations

from typing import Any, Optional, cast

from slangpy.core.native import AccessType, Shape, CallMode

from slangpy.reflection.reflectiontypes import is_matching_array_type, VectorType
from slangpy.types.tensor import Tensor
from slangpy.types.buffer import innermost_type
from slangpy.core.native import NativeTensorMarshall, NativeTensor

from slangpy import TypeReflection, ShaderObject, ShaderCursor, BufferUsage
from slangpy.reflection import (
    TYPE_OVERRIDES,
    SlangProgramLayout,
    SlangType,
    TypeReflection,
    ArrayType,
    ScalarType,
    MatrixType,
    UnknownType,
    InterfaceType,
    ITensorType,
    TensorType,
    vectorize_type,
    EXPERIMENTAL_VECTORIZATION,
)
from slangpy.bindings import (
    PYTHON_TYPES,
    BindContext,
    BoundVariable,
    CodeGenBlock,
    ReturnContext,
)
from slangpy.builtin.ndbuffer import get_ndbuffer_marshall_type
import slangpy.reflection.vectorize as spyvec


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

        slang_type = layout.tensor_type(
            element_type=element_type,
            dims=dims,
            writable=writable,
            tensor_type=TensorType.tensor,
            has_grad_in=d_in is not None,
            has_grad_out=d_out is not None,
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
            if bound_type.has_grad_in and self.d_in is None:
                raise ValueError("Can't pass tensor without input gradient to one that requires it")
            if bound_type.has_grad_out and self.d_out is None:
                raise ValueError(
                    "Can't pass tensor without output gradient to one that requires it"
                )

            tensor_type = (
                TensorType.tensor
                if bound_type.tensor_type == TensorType.interface
                else bound_type.tensor_type
            )

            return self.layout.tensor_type(
                element_type=bound_type.dtype,
                dims=bound_type.dims,
                writable=bound_type.writable,
                tensor_type=tensor_type,
                has_grad_in=bound_type.has_grad_in,
                has_grad_out=bound_type.has_grad_out,
            )

        # if implicit element casts enabled, allow conversion from type to element type
        if types_equal(self.slang_element_type, bound_type):
            return bound_type
        if is_matching_array_type(bound_type, self.slang_element_type):
            return self.slang_element_type

        # if implicit tensor casts enabled, allow conversion from vector to element type
        # or array type
        # Be lenient and allow e.g. passing float to float[N] or float[M][N]
        if types_equal(self.slang_element_type, innermost_type(bound_type)):
            if is_nested_array(bound_type) and len(bound_type.shape) <= 2:
                return bound_type
            if isinstance(bound_type, VectorType):
                return bound_type

        # Default to casting to itself
        return self.slang_type

    def resolve_types(self, context: BindContext, bound_type: SlangType):
        self_element_type = cast(SlangType, self.slang_element_type)
        self_dims = self.dims
        self_writable = self.writable

        # Trying to pass tensor to tensor - handle programmatically
        if isinstance(bound_type, ITensorType):
            if bound_type.writable and not self.writable:
                return None
            if bound_type.has_grad_in and self.d_in is None:
                return None
            if bound_type.has_grad_out and self.d_out is None:
                return None

            if bound_type.tensor_type == TensorType.interface:
                tensor_type = TensorType.tensor
                has_grad_in = context.call_mode == CallMode.bwds and self.d_in is not None
                has_grad_out = context.call_mode == CallMode.bwds and self.d_out is not None
            else:
                tensor_type = bound_type.tensor_type
                has_grad_in = bound_type.has_grad_in
                has_grad_out = bound_type.has_grad_out

            bound_element_type = bound_type.element_type
            if isinstance(bound_element_type, UnknownType) or bound_element_type.is_generic:
                el_type = self_element_type
            else:
                el_type = bound_element_type
            if bound_type.dims == 0:
                dims = self_dims
            else:
                dims = bound_type.dims
            if not types_equal(el_type, self_element_type):
                return None

            return [
                self.layout.tensor_type(
                    element_type=el_type,
                    dims=dims,
                    writable=bound_type.writable,
                    tensor_type=tensor_type,
                    has_grad_in=has_grad_in,
                    has_grad_out=has_grad_out,
                )
            ]

        # If target type is fully generic, always add tensor type as option
        if isinstance(bound_type, (UnknownType, InterfaceType)):
            results: list[SlangType] = []
            results.append(self.slang_type)
            results.append(self.slang_element_type)
            return results

        if EXPERIMENTAL_VECTORIZATION:
            # Ambiguous case that vectorizer in slang cannot resolve on its own - could be element type or array of element type
            # Add both options, and rely on later slang specialization to pick the correct one (or identify it as genuinely ambiguous)
            if (
                isinstance(self_element_type, ArrayType)
                and isinstance(bound_type, ArrayType)
                and isinstance(bound_type.element_type, UnknownType)
            ):
                results: list[SlangType] = []
                results.append(self_element_type)
                results.append(
                    context.layout.require_type_by_name(
                        f"{self_element_type.full_name}[{bound_type.num_elements}]"
                    )
                )
                return results
            marshall = get_ndbuffer_marshall_type(
                context, self_element_type, self_writable, self_dims
            )
            specialized = vectorize_type(marshall, bound_type)
            return [specialized]

        # Match element type exactly
        if self_element_type.full_name == bound_type.full_name:
            return [self_element_type]

        # Match buffer container types
        as_structuredbuffer_type = spyvec.container_to_structured_buffer(
            self_element_type, self_writable, bound_type
        )
        if as_structuredbuffer_type is not None:
            return [as_structuredbuffer_type]
        as_byteaddressbuffer_type = spyvec.container_to_byte_address_buffer(
            self_element_type, self_writable, bound_type
        )
        if as_byteaddressbuffer_type is not None:
            return [as_byteaddressbuffer_type]

        # Match pointers
        as_pointer = spyvec.container_to_pointer(self_element_type, bound_type)
        if as_pointer is not None:
            return [as_pointer]

        # NDBuffer of scalars can load matrices of known size
        as_matrix = spyvec.scalar_to_sized_matrix(self_element_type, bound_type)
        if as_matrix is not None:
            return [as_matrix]

        # NDBuffer of scalars can load vectors of known size
        as_vector = spyvec.scalar_to_sized_vector(self_element_type, bound_type)
        if as_vector is not None:
            return [as_vector]

        # Handle ambiguous case vectorizing against generic array type
        as_generic_array_candidates = spyvec.container_to_generic_array_candidates(
            self_element_type, bound_type
        )
        if as_generic_array_candidates is not None:
            return as_generic_array_candidates

        # NDBuffer of elements can load higher dimensional arrays of known size
        as_sized_array = spyvec.container_to_sized_array(self_element_type, bound_type, self_dims)
        if as_sized_array is not None:
            return [as_sized_array]

        # Support resolving generic struct
        as_struct = spyvec.struct_to_struct(self_element_type, bound_type)
        if as_struct is not None:
            return [as_struct]

        # Support resolving generic array
        as_array = spyvec.array_to_array(self_element_type, bound_type)
        if as_array is not None:
            return [as_array]

        # Support resolving generic matrix
        as_matrix = spyvec.matrix_to_matrix(self_element_type, bound_type)
        if as_matrix is not None:
            return [as_matrix]

        # Support resolving generic vector
        as_vector = spyvec.vector_to_vector(self_element_type, bound_type)
        if as_vector is not None:
            return [as_vector]

        # Support resolving generic scalar
        as_scalar = spyvec.scalar_to_scalar(self_element_type, bound_type)
        if as_scalar is not None:
            return [as_scalar]
        return None

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
            type_name = ITensorType.build_tensor_name(
                element_type=self.slang_element_type,
                dims=self.dims,
                writable=binding.vector_type.writable,
                tensor_type=binding.vector_type.tensor_type,
                has_grad_in=binding.vector_type.has_grad_in,
                has_grad_out=binding.vector_type.has_grad_out,
            )
        else:
            writable = binding.access[0] in (AccessType.write, AccessType.readwrite)
            type_name = ITensorType.build_tensor_name(
                element_type=self.slang_element_type,
                dims=self.dims,
                writable=writable,
                tensor_type=TensorType.tensor,
                has_grad_in=self.d_in is not None,
                has_grad_out=self.d_out is not None,
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
