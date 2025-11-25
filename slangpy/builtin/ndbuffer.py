# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Any, Optional, Union, cast

from slangpy.core.enums import PrimType
from slangpy.core.native import (
    AccessType,
    CallContext,
    Shape,
    CallMode,
    NativeNDBuffer,
    NativeNDBufferMarshall,
)

from slangpy import BufferUsage, TypeReflection, ShaderCursor, ShaderObject
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
    TYPE_OVERRIDES,
    SlangProgramLayout,
    SlangType,
    ScalarType,
    VectorType,
    MatrixType,
    StructuredBufferType,
    PointerType,
    ArrayType,
    UnknownType,
    InterfaceType,
    ITensorType,
    is_matching_array_type,
    is_unknown,
    is_known,
    vectorize_type,
    EXPERIMENTAL_VECTORIZATION,
)
from slangpy.types import NDBuffer
from slangpy.experimental.diffbuffer import NDDifferentiableBuffer
import slangpy.reflection.vectorize as spyvec


class StopDebuggerException(Exception):
    pass


def _calc_broadcast(context: CallContext, binding: BoundVariableRuntime):
    broadcast = []
    transform = cast(Shape, binding.transform)
    for i in range(len(transform)):
        csidx = transform[i]
        broadcast.append(context.call_shape[csidx] != binding.shape[i])
    broadcast.extend([False] * (len(binding.shape) - len(broadcast)))
    return broadcast


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


def get_ndbuffer_type(
    context: BindContext, element_type: SlangType, writable: bool, dims: int
) -> SlangType:
    prefix = "RW" if writable else ""
    type_name = f"{prefix}NDBuffer<{element_type.full_name},{dims}>"
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
        buffer_type = get_ndbuffer_type(context, self_element_type, self_writable, self_dims)
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
        return [get_ndbuffer_type(context, el_type, bound_type.writable, dims)]

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
            cgb.type_alias(f"_t_{name}", f"RWNDBuffer<{et.full_name},{self.dims}>")
        else:
            cgb.type_alias(f"_t_{name}", f"NDBuffer<{et.full_name},{self.dims}>")


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
            f"{prefix}NDBuffer<{self.slang_element_type.full_name},{self.dims}>"
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
            f"{prefix}NDBuffer<{slang_el_type.full_name},{dims}>"
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


def generate_differential_buffer(
    name: str,
    context: str,
    primal_storage: str,
    deriv_storage: str,
    primal_target: str,
    deriv_target: Optional[str],
):
    assert primal_storage
    assert deriv_storage
    assert primal_target
    if deriv_target is None:
        deriv_target = primal_target

    DIFF_PAIR_CODE = f"""
struct _t_{name}
{{
    {primal_storage} primal;
    {deriv_storage} derivative;

    [Differentiable, BackwardDerivative(load_bwd)]
    void load({context} context, out {primal_target} value) {{ primal.load(context, value); }}
    void load_bwd({context} context, {deriv_target} value) {{ derivative.store(context, value); }}

    [Differentiable, BackwardDerivative(store_bwd)]
    void store({context} context, {primal_target} value) {{ primal.store(context, value); }}
    void store_bwd({context} context, inout DifferentialPair<{primal_target}> value) {{
        {deriv_target} grad;
        derivative.load(context, grad);
        value = diffPair(value.p, grad);
    }}
}}
"""
    return DIFF_PAIR_CODE


class NDDifferentiableBufferMarshall(BaseNDBufferMarshall):

    def __init__(
        self,
        layout: SlangProgramLayout,
        element_type: SlangType,
        dims: int,
        writable: bool,
    ):
        super().__init__(layout, element_type, dims, writable)

        if not element_type.differentiable:
            raise ValueError(f"Elements of differentiable buffer must be differentiable.")

    @property
    def has_derivative(self) -> bool:
        return True

    def reduce_type(self, context: BindContext, dimensions: int):
        return ndbuffer_reduce_type(self, context, dimensions)

    def resolve_type(self, context: BindContext, bound_type: "SlangType"):
        return ndbuffer_resolve_type(self, context, bound_type)

    def resolve_dimensionality(
        self,
        context: BindContext,
        binding: BoundVariable,
        vector_target_type: "SlangType",
    ):
        return ndbuffer_resolve_dimensionality(self, context, binding, vector_target_type)

    # Call data can only be read access to primal, and simply declares it as a variable
    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: "BoundVariable"):
        access = binding.access
        name = binding.variable_name

        if isinstance(binding.vector_type, ITensorType):
            # If passing to NDBuffer, just use the NDBuffer type
            assert access[0] == AccessType.read
            assert isinstance(binding.vector_type, ITensorType)
            cgb.type_alias(f"_t_{name}", binding.vector_type.full_name)
        else:

            if (
                context.call_mode != CallMode.prim
                and self.writable
                and access[0] == AccessType.none
            ):
                access = (AccessType.write, access[1])

            # If broadcasting to an element, use full diff pair logic
            prim_el = self.slang_element_type.full_name
            deriv_el = prim_el + ".Differential"
            dim = self.dims

            if access[0] == AccessType.none:
                primal_storage = f"NoneType"
            elif access[0] == AccessType.read:
                primal_storage = f"NDBuffer<{prim_el},{dim}>"
            else:
                primal_storage = f"RWNDBuffer<{prim_el},{dim}>"

            if access[1] == AccessType.none:
                deriv_storage = f"NoneType"
            elif access[1] == AccessType.read:
                deriv_storage = f"NDBuffer<{deriv_el},{dim}>"
            else:
                deriv_storage = f"RWNDBuffer<{deriv_el},{dim}>"

            assert binding.vector_type is not None
            primal_target = binding.vector_type.full_name
            deriv_target = binding.vector_type.full_name + ".Differential"

            slang_context = f"ContextND<{binding.call_dimensionality}>"

            cgb.append_code_indented(
                generate_differential_buffer(
                    name,
                    slang_context,
                    primal_storage,
                    deriv_storage,
                    primal_target,
                    deriv_target,
                )
            )

    def create_calldata(
        self,
        context: CallContext,
        binding: "BoundVariableRuntime",
        data: NDDifferentiableBuffer,
    ) -> Any:
        if isinstance(binding.vector_type, ITensorType):
            return {
                "buffer": data.storage,
                "_shape": data.shape.as_tuple(),
                "layout": {"strides": data.strides, "offset": data.offset},
            }
        else:
            broadcast = _calc_broadcast(context, binding)
            access = binding.access
            assert binding.transform is not None
            res = {}
            for prim in PrimType:
                prim_name = prim.name
                prim_access = access[prim.value]
                if prim_access != AccessType.none:
                    ndbuffer = data if prim == PrimType.primal else data.grad
                    assert ndbuffer is not None
                    value = ndbuffer.storage if prim == PrimType.primal else ndbuffer.storage
                    res[prim_name] = {
                        "buffer": value,
                        "_shape": ndbuffer.shape.as_tuple(),
                        "layout": {
                            "strides": [
                                ndbuffer.strides[i] if not broadcast[i] else 0
                                for i in range(len(ndbuffer.strides))
                            ],
                            "offset": ndbuffer.offset,
                        },
                    }
            return res

    def create_output(self, context: CallContext, binding: BoundVariableRuntime) -> Any:
        return NDDifferentiableBuffer(
            context.device,
            self.slang_element_type,
            shape=context.call_shape,
            requires_grad=True,
            usage=BufferUsage.shader_resource | BufferUsage.unordered_access,
        )

    def read_output(
        self,
        context: CallContext,
        binding: BoundVariableRuntime,
        data: NDDifferentiableBuffer,
    ) -> Any:
        return data

    def create_dispatchdata(self, data: NDDifferentiableBuffer) -> Any:
        return data.uniforms()

    def get_shape(self, value: Optional[NDBuffer] = None) -> Shape:
        if value is not None:
            return value.shape + self.slang_element_type.shape
        else:
            return Shape((-1,) * self.dims) + self.slang_element_type.shape

    @property
    def is_writable(self) -> bool:
        return self.writable


def create_gradvr_type_for_value(layout: SlangProgramLayout, value: Any):
    if isinstance(value, NDDifferentiableBuffer):
        return NDDifferentiableBufferMarshall(
            layout,
            value.dtype,
            len(value.shape),
            (value.usage & BufferUsage.unordered_access) != 0,
        )
    elif isinstance(value, ReturnContext):
        return NDDifferentiableBufferMarshall(
            layout, value.slang_type, value.bind_context.call_dimensionality, True
        )
    else:
        raise ValueError(
            f"Unexpected type {type(value)} attempting to create NDDifferentiableBuffer marshall"
        )


PYTHON_TYPES[NDDifferentiableBuffer] = create_gradvr_type_for_value
