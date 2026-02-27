# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional, Protocol
from slangpy.bindings import BoundVariable, BindContext, CodeGenBlock
from slangpy.core.native import CallMode, AccessType
from slangpy.reflection import (
    SlangType,
    ITensorType,
    TensorType,
    TensorViewType,
    DiffTensorViewType,
    ArrayType,
    InterfaceType,
    UnknownType,
    SlangProgramLayout,
    vectorize_type,
    EXPERIMENTAL_VECTORIZATION,
    ResourceType,
    TensorAccess,
    VectorType,
)
import slangpy.reflection.vectorize as spyvec


class ITensorMarshall(Protocol):
    """
    Protocol for type marshalling of any container that behaves as a tensor.
    """

    @property
    def dims(self) -> int: ...
    @property
    def writable(self) -> bool: ...
    @property
    def slang_element_type(self) -> SlangType: ...
    @property
    def slang_type(self) -> SlangType: ...
    @property
    def layout(self) -> SlangProgramLayout: ...
    @property
    def d_in(self) -> Optional["ITensorMarshall"]: ...
    @property
    def d_out(self) -> Optional["ITensorMarshall"]: ...


def types_equal(a: SlangType, b: SlangType):
    # TODO: Exact comparison of slang types is not currently possible, and we do the next closest thing
    # of comparing their fully qualified names. This will give false positives on types from different
    # modules but with the same name, and false negatives on the same type with different names
    # (e.g. via typedef)
    return a.full_name == b.full_name


def build_tensor_marshall_type(
    context: BindContext, element_type: SlangType, writable: bool, dims: int
) -> SlangType:
    """
    Used for experimental vectorization support - gets the appropriate TensorMarshall type
    for the given element type, writability, and number of dimensions.
    """
    type_name = f"TensorMarshall<{element_type.full_name},{dims},{'true' if writable else 'false'}>"
    slang_type = context.layout.find_type_by_name(type_name)
    if slang_type is None:
        raise ValueError(f"Could not find type {type_name} in program layout")
    return slang_type


def resolve_types(self: ITensorMarshall, context: BindContext, bound_type: SlangType):
    """
    During vectorizing, this function is called to match up the Python argument type being
    passed (represented by the marshall) to the Slang paramameter type being bound to.
    This function returns a list of possible Slang types that the argument can be converted to
    in order to satisfy the parameter type.

    If no valid conversions are possible, this function can return None, or raise a type
    error with a descriptive message.
    """

    self_type = self.slang_type
    self_element_type = self.slang_element_type
    self_dims = self.dims
    self_writable = self.writable

    if isinstance(bound_type, TensorViewType):
        tensorview_element = bound_type.dtype

        # If TensorView has generic type (Unknown), use tensor's element type
        if isinstance(tensorview_element, UnknownType) or tensorview_element.is_generic:
            resolved_element = self_element_type
        elif not types_equal(self_element_type, tensorview_element):
            # Allow scalar tensor dtype to bind to TensorView<VectorType>
            # e.g., float32 tensor -> TensorView<float2>
            if isinstance(tensorview_element, VectorType) and types_equal(
                self_element_type, tensorview_element.scalar_type
            ):
                resolved_element = tensorview_element
            else:
                raise TypeError(
                    f"Cannot bind tensor with dtype {self_element_type.full_name} "
                    f"to TensorView<{tensorview_element.full_name}>"
                )
        else:
            resolved_element = tensorview_element

        tensorview_type = self.layout.tensorview_type(resolved_element)
        if tensorview_type is None:
            raise ValueError(f"TensorView<{resolved_element.full_name}> not found")
        return [tensorview_type]

    if isinstance(bound_type, DiffTensorViewType):
        dtv_element = bound_type.dtype

        # If DiffTensorView has generic type (Unknown), use tensor's element type
        if isinstance(dtv_element, UnknownType) or dtv_element.is_generic:
            resolved_element = self_element_type
        elif not types_equal(self_element_type, dtv_element):
            # Allow scalar tensor dtype to bind to DiffTensorView<VectorType>
            # e.g., float32 tensor -> DiffTensorView<float2>
            if isinstance(dtv_element, VectorType) and types_equal(
                self_element_type, dtv_element.scalar_type
            ):
                resolved_element = dtv_element
            else:
                raise TypeError(
                    f"Cannot bind tensor with dtype {self_element_type.full_name} "
                    f"to DiffTensorView<{dtv_element.full_name}>"
                )
        else:
            resolved_element = dtv_element

        dtv_type = self.layout.difftensorview_type(resolved_element)
        if dtv_type is None:
            raise ValueError(f"DiffTensorView<{resolved_element.full_name}> not found")
        return [dtv_type]

    # Trying to pass tensor to tensor - handle programmatically
    if isinstance(bound_type, ITensorType):
        if bound_type.writable and not self.writable:
            raise TypeError(
                f"Can't pass a read-only tensor to a writable tensor ({bound_type.full_name})"
            )

        # Gradients need binding if using a DiffTensor, or an IDiffTensor in non-primitive pass
        grads_used = bound_type.tensor_type == TensorType.difftensor or (
            bound_type.tensor_type == TensorType.idifftensor and context.call_mode != CallMode.prim
        )
        if grads_used:
            if bound_type.has_grad_in and self.d_in is None:
                raise TypeError(
                    f"Can't pass tensor without input gradient to one that requires it ({bound_type.full_name})"
                )
            if bound_type.has_grad_out and self.d_out is None:
                raise TypeError(
                    f"Can't pass tensor without output gradient to one that requires it ({bound_type.full_name})"
                )

        # Select appropriate tensor type:
        # ITensor -> Tensor
        # IDiffTensor -> PrimalTensor (primal pass) or DiffTensor (bwd/fwd diff pass)
        # Other tensor types map directly to the bound type
        if bound_type.tensor_type == TensorType.itensor:
            tensor_type = TensorType.tensor
        elif bound_type.tensor_type == TensorType.idifftensor:
            if context.call_mode == CallMode.prim:
                tensor_type = TensorType.primaltensor
            else:
                tensor_type = TensorType.difftensor
        else:
            tensor_type = bound_type.tensor_type

        # Element type is taken from the marshall if the target is unknown or generic.
        bound_element_type = bound_type.element_type
        if isinstance(bound_element_type, UnknownType) or bound_element_type.is_generic:
            el_type = self_element_type
        else:
            el_type = bound_element_type

        # Dimensions - if target is 0-dim (fully generic), take from marshall
        if bound_type.dims == 0:
            dims = self_dims
        else:
            dims = bound_type.dims

        if not types_equal(el_type, self_element_type):
            raise TypeError(
                f"Can't convert tensor with element type {self_element_type.full_name} to tensor with element type {el_type.full_name} ({bound_type.full_name})"
            )

        return [
            self.layout.tensor_type(
                element_type=el_type,
                dims=dims,
                access=bound_type.access,
                tensor_type=tensor_type,
            )
        ]

    # If target type is fully generic, always add tensor type as option
    if isinstance(bound_type, (UnknownType, InterfaceType)):
        results: list[SlangType] = []
        results.append(self_type)
        results.append(self_element_type)
        return results

    # Experimental vectorization system attempts to use a special slang type that represents the
    # marshall itself to resolve against rules defined in Slang.
    if EXPERIMENTAL_VECTORIZATION:
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
        marshall = build_tensor_marshall_type(context, self_element_type, self_writable, self_dims)
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

    # Tensor of scalars can load matrices of known size
    as_matrix = spyvec.scalar_to_sized_matrix(self_element_type, bound_type)
    if as_matrix is not None:
        return [as_matrix]

    # Tensor of scalars can load vectors of known size
    as_vector = spyvec.scalar_to_sized_vector(self_element_type, bound_type)
    if as_vector is not None:
        return [as_vector]

    # Handle ambiguous case vectorizing against generic array type
    as_generic_array_candidates = spyvec.container_to_generic_array_candidates(
        self_element_type, bound_type
    )
    if as_generic_array_candidates is not None:
        return as_generic_array_candidates

    # Tensor of elements can load higher dimensional arrays of known size
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


def reduce_type(self: ITensorMarshall, context: BindContext, dimensions: int):
    """
    During vectorizing, if the target slang parameter type is not known, but through explicit
    mapping its the desired dimensionality is known, this function attempts to get
    a candidate type based purely on chopping off dimensions.
    """
    if dimensions == 0:
        return self.slang_type
    elif dimensions == self.dims:
        return self.slang_element_type
    elif dimensions < self.dims:
        # Not sure how to handle this yet - what do we want if reducing by some dimensions
        # Should this return a smaller buffer? How does that end up being cast to, eg, vector.
        # By returning None, this just falls back to the slang argument type.
        return None
    else:
        raise ValueError("Cannot reduce dimensions of Tensor")


def resolve_dimensionality(
    self: ITensorMarshall,
    context: BindContext,
    binding: BoundVariable,
    vector_target_type: "SlangType",
):
    """
    Once a target type has been selected for vectorization, this function is called
    to determine the dimensionality of the call from the perspective of a bound variable.
    """
    if isinstance(vector_target_type, TensorViewType):
        return 0
    if isinstance(vector_target_type, DiffTensorViewType):
        return 0
    if isinstance(vector_target_type, ITensorType):
        return self.dims - vector_target_type.dims
    else:
        return self.dims + len(self.slang_element_type.shape) - len(vector_target_type.shape)


def gen_calldata(
    self: ITensorMarshall, cgb: CodeGenBlock, context: BindContext, binding: BoundVariable
):
    if isinstance(binding.vector_type, ITensorType):
        type_name = ITensorType.build_tensor_name(
            element_type=self.slang_element_type,
            dims=self.dims,
            access=binding.vector_type.access,
            tensor_type=binding.vector_type.tensor_type,
        )
    elif isinstance(binding.vector_type, TensorViewType):
        type_name = TensorViewType.build_tensorview_name(binding.vector_type.dtype)
    elif isinstance(binding.vector_type, DiffTensorViewType):
        type_name = DiffTensorViewType.build_difftensorview_name(binding.vector_type.dtype)
    else:
        if isinstance(binding.vector_type, ResourceType):
            access = (
                TensorAccess.read if not binding.vector_type.writable else TensorAccess.read_write
            )
        elif context.call_mode == CallMode.prim or binding.access[0] != AccessType.none:
            if binding.access[0] == AccessType.read:
                access = TensorAccess.read
            elif binding.access[0] == AccessType.write:
                access = TensorAccess.write
            else:
                access = TensorAccess.read_write
        else:
            if binding.access[1] == AccessType.read:
                access = TensorAccess.write
            elif binding.access[1] == AccessType.write:
                access = TensorAccess.read
            else:
                access = TensorAccess.read_write

        if context.call_mode == CallMode.prim or binding.access[1] == AccessType.none:
            tensor_type = TensorType.tensor
        else:
            tensor_type = TensorType.difftensor

        type_name = ITensorType.build_tensor_name(
            element_type=self.slang_element_type,
            dims=self.dims,
            access=access,
            tensor_type=tensor_type,
        )
    cgb.type_alias(f"_t_{binding.variable_name}", type_name)


def gen_trampoline_load(
    self: ITensorMarshall, cgb: CodeGenBlock, binding: BoundVariable, is_entry_point: bool
) -> bool:
    if not isinstance(binding.vector_type, (TensorViewType, DiffTensorViewType)):
        return False
    if is_entry_point:
        data_name = f"__calldata__.{binding.variable_name}"
    else:
        data_name = f"call_data.{binding.variable_name}"
    cgb.append_statement(f"{binding.variable_name} = {data_name}")
    return True


def gen_trampoline_store(
    self: ITensorMarshall, cgb: CodeGenBlock, binding: BoundVariable, is_entry_point: bool
) -> bool:
    if not isinstance(binding.vector_type, (TensorViewType, DiffTensorViewType)):
        return False
    return True
