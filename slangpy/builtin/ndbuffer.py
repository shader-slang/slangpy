# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Any, Union, cast

from slangpy.core.native import (
    AccessType,
    CallContext,
    Shape,
    NativeNDBuffer,
    NativeNDBufferMarshall,
)

from slangpy import BufferUsage, ShaderCursor, ShaderObject
from slangpy.bindings import (
    PYTHON_TYPES,
    Marshall,
    BindContext,
    BoundVariable,
    BoundVariableRuntime,
    CodeGenBlock,
    ReturnContext,
)
from slangpy.reflection import (
    SlangProgramLayout,
    SlangType,
    VectorType,
    MatrixType,
    StructuredBufferType,
    PointerType,
    ArrayType,
    UnknownType,
    InterfaceType,
    ITensorType,
    TensorAccess,
    TensorType,
    is_matching_array_type,
    vectorize_type,
    EXPERIMENTAL_VECTORIZATION,
)
from slangpy.types import NDBuffer
import slangpy.reflection.vectorize as spyvec


class StopDebuggerException(Exception):
    pass


def ndbuffer_reduce_type(
    self: Union[NativeNDBufferMarshall, "BaseNDBufferMarshall"],
    context: BindContext,
    dimensions: int,
):
    if dimensions == 0:
        return self.slang_type
    elif dimensions == self.dims:
        return self.slang_element_type
    elif dimensions < self.dims:
        # Not sure how to handle this yet - what do we want if reducing by some dimensions
        # Should this return a smaller buffer? How does that end up being cast to, eg, vector.
        return None
    else:
        raise ValueError("Cannot reduce dimensions of NDBuffer")


def ndbuffer_resolve_type(
    self: Union[NativeNDBufferMarshall, "BaseNDBufferMarshall"],
    context: BindContext,
    bound_type: "SlangType",
):

    if isinstance(bound_type, ITensorType) or isinstance(bound_type, StructuredBufferType):
        # If the bound type is an NDBuffer, verify properties match then just use it
        if bound_type.writable and not self.writable:
            raise ValueError("Attempted to bind a writable buffer to a read-only buffer")
        if bound_type.element_type != self.slang_element_type:
            raise ValueError("Attempted to bind a buffer with a different element type")
        if isinstance(bound_type, StructuredBufferType) and self.dims != 1:
            raise ValueError("Attempted to pass an NDBuffer that is not 1D to a StructuredBuffer")
        return bound_type

    # if implicit element casts enabled, allow conversion from type to element type
    if self.slang_element_type == bound_type:
        return bound_type
    if is_matching_array_type(bound_type, cast(SlangType, self.slang_element_type)):
        return self.slang_element_type
    # This is such a common conversion with numpy 64 bit arrays to ptrs that we handle it explicitly
    # TODO: Use host casting test instead?
    if self.slang_element_type.full_name == "uint64_t" and isinstance(bound_type, PointerType):
        return bound_type

    # if implicit tensor casts enabled, allow conversion from vector to element type
    if (
        isinstance(bound_type, VectorType) or isinstance(bound_type, MatrixType)
    ) and self.slang_element_type == bound_type.scalar_type:
        return bound_type

    # Default to just casting to itself (i.e. no implicit cast)
    return self.slang_type


def get_ndbuffer_marshall_type(
    context: BindContext, element_type: SlangType, writable: bool, dims: int
) -> SlangType:
    type_name = (
        f"NDBufferMarshall<{element_type.full_name},{dims},{'true' if writable else 'false'}>"
    )
    slang_type = context.layout.find_type_by_name(type_name)
    if slang_type is None:
        raise ValueError(f"Could not find type {type_name} in program layout")
    return slang_type


def get_structuredbuffer_type(
    context: BindContext, element_type: SlangType, writable: bool
) -> SlangType:
    prefix = "RW" if writable else ""
    type_name = f"{prefix}StructuredBuffer<{element_type.full_name}>"
    slang_type = context.layout.find_type_by_name(type_name)
    if slang_type is None:
        raise ValueError(f"Could not find type {type_name} in program layout")
    return slang_type


def ndbuffer_resolve_types(
    self: Union[NativeNDBufferMarshall, "BaseNDBufferMarshall"],
    context: BindContext,
    bound_type: "SlangType",
):

    self_element_type = cast(SlangType, self.slang_element_type)
    self_dims = self.dims
    self_writable = self.writable
    results: list[SlangType] = []

    # If target type is fully generic, allow buffer or element type
    if isinstance(bound_type, (UnknownType, InterfaceType)):
        buffer_type = bound_type.program.tensor_type(
            self_element_type,
            self_dims,
            TensorAccess.read_write if self_writable else TensorAccess.read,
            TensorType.tensor,
        )
        results.append(buffer_type)
        results.append(self_element_type)
        return results

    # Otherwise, attempt to use slang's typing system to map the bound type to the marshall
    if EXPERIMENTAL_VECTORIZATION:
        # Ambiguous case that vectorizer in slang cannot resolve on its own - could be element type or array of element type
        # Add both options, and rely on later slang specialization to pick the correct one (or identify it as genuinely ambiguous)
        if isinstance(bound_type, ArrayType) and isinstance(bound_type.element_type, UnknownType):
            if bound_type.num_dims >= 0:
                results.append(self_element_type)
            if bound_type.num_dims >= 1 and bound_type.shape[0] >= 1:
                results.append(
                    context.layout.require_type_by_name(
                        f"{self_element_type.full_name}[{bound_type.shape[0]}]"
                    )
                )
            if bound_type.num_dims >= 2 and bound_type.shape[0] >= 1 and bound_type.shape[1] >= 1:
                results.append(
                    context.layout.require_type_by_name(
                        f"{self_element_type.full_name}[{bound_type.shape[0]}][{bound_type.shape[1]}]"
                    )
                )
            return results

        marshall = get_ndbuffer_marshall_type(context, self_element_type, self_writable, self_dims)
        specialized = vectorize_type(marshall, bound_type)
        if specialized is not None:
            results.append(specialized)

    # Target type is NDBuffer
    if isinstance(bound_type, ITensorType):
        if bound_type.writable and not self_writable:
            return None
        bound_element_type = bound_type.element_type
        if isinstance(bound_element_type, UnknownType) or bound_element_type.is_generic:
            el_type = self_element_type
        else:
            el_type = bound_element_type
        if bound_type.dims == 0:
            dims = self_dims
        else:
            dims = bound_type.dims
        if el_type.full_name != self_element_type.full_name:
            return None
        return [
            bound_type.program.tensor_type(
                el_type,
                dims,
                TensorAccess.read_write if bound_type.writable else TensorAccess.read,
                TensorType.tensor,
            )
        ]

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


def ndbuffer_resolve_dimensionality(
    self: Union[NativeNDBufferMarshall, "BaseNDBufferMarshall"],
    context: BindContext,
    binding: BoundVariable,
    vector_target_type: "SlangType",
):
    return self.dims + len(self.slang_element_type.shape) - len(vector_target_type.shape)


def ndbuffer_gen_calldata(
    self: Union[NativeNDBufferMarshall, "BaseNDBufferMarshall"],
    cgb: CodeGenBlock,
    context: BindContext,
    binding: "BoundVariable",
):
    access = binding.access
    name = binding.variable_name
    assert access[0] != AccessType.none
    assert access[1] == AccessType.none
    writable = access[0] != AccessType.read
    if isinstance(binding.vector_type, ITensorType):
        # If passing to NDBuffer, just use the NDBuffer type
        assert access[0] == AccessType.read
        assert isinstance(binding.vector_type, ITensorType)
        cgb.type_alias(f"_t_{name}", binding.vector_type.full_name)
    else:
        # If we pass to a structured buffer, check the writable flag from the type
        if isinstance(binding.vector_type, StructuredBufferType):
            writable = binding.vector_type.writable

        # If broadcasting to an element, use the type of this buffer for code gen\
        et = cast(SlangType, self.slang_element_type)
        if writable:
            cgb.type_alias(f"_t_{name}", f"RWTensor<{et.full_name},{self.dims}>")
        else:
            cgb.type_alias(f"_t_{name}", f"Tensor<{et.full_name},{self.dims}>")


class BaseNDBufferMarshall(Marshall):
    def __init__(
        self,
        layout: SlangProgramLayout,
        element_type: SlangType,
        dims: int,
        writable: bool,
    ):
        super().__init__(layout)

        self.dims = dims
        self.writable = writable

        prefix = "RW" if self.writable else ""

        # Note: find by name handles the fact that element type may not be from the same program layout
        slet = layout.find_type_by_name(element_type.full_name)
        assert slet is not None
        self.slang_element_type = slet

        slt = layout.find_type_by_name(
            f"{prefix}Tensor<{self.slang_element_type.full_name},{self.dims}>"
        )
        assert slt is not None
        self.slang_type = slt


class NDBufferMarshall(NativeNDBufferMarshall):

    def __init__(
        self,
        layout: SlangProgramLayout,
        element_type: SlangType,
        dims: int,
        writable: bool,
    ):

        slang_el_type = layout.find_type_by_name(element_type.full_name)
        assert slang_el_type is not None

        slang_el_layout = slang_el_type.buffer_layout

        prefix = "RW" if writable else ""
        slang_buffer_type = layout.find_type_by_name(
            f"{prefix}Tensor<{slang_el_type.full_name},{dims}>"
        )
        assert slang_buffer_type is not None

        super().__init__(
            dims, writable, slang_buffer_type, slang_el_type, slang_el_layout.reflection
        )

    def __repr__(self) -> str:
        return f"NDBuffer[dtype={self.slang_element_type.full_name}, dims={self.dims}, writable={self.writable}]"

    @property
    def is_writable(self) -> bool:
        return self.writable

    def reduce_type(self, context: BindContext, dimensions: int):
        return ndbuffer_reduce_type(self, context, dimensions)

    def resolve_type(self, context: BindContext, bound_type: "SlangType"):
        return ndbuffer_resolve_type(self, context, bound_type)

    def resolve_types(self, context: BindContext, bound_type: "SlangType"):
        return ndbuffer_resolve_types(self, context, bound_type)

    def resolve_dimensionality(
        self,
        context: BindContext,
        binding: BoundVariable,
        vector_target_type: "SlangType",
    ):
        return ndbuffer_resolve_dimensionality(self, context, binding, vector_target_type)

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: "BoundVariable"):
        return ndbuffer_gen_calldata(self, cgb, context, binding)

    def build_shader_object(self, context: "BindContext", data: Any) -> "ShaderObject":
        et = cast(SlangType, self.slang_element_type)
        slang_type = context.layout.find_type_by_name(f"RWNDBuffer<{et.full_name},{self.dims}>")
        so = context.device.create_shader_object(slang_type.uniform_layout.reflection)
        cursor = ShaderCursor(so)
        cursor.write(data.uniforms())
        return so


def create_vr_type_for_value(layout: SlangProgramLayout, value: Any):
    if isinstance(value, (NDBuffer, NativeNDBuffer)):
        return NDBufferMarshall(
            layout,
            cast(SlangType, value.dtype),
            len(value.shape),
            (value.usage & BufferUsage.unordered_access) != 0,
        )
    elif isinstance(value, ReturnContext):
        return NDBufferMarshall(
            layout, value.slang_type, value.bind_context.call_dimensionality, True
        )
    else:
        raise ValueError(f"Unexpected type {type(value)} attempting to create NDBuffer marshall")


PYTHON_TYPES[NativeNDBuffer] = create_vr_type_for_value
PYTHON_TYPES[NDBuffer] = create_vr_type_for_value
