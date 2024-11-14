

from typing import Any, Optional, cast

from kernelfunctions.bindings.valuetype import slang_type_to_return_type
from kernelfunctions.core import CodeGenBlock, BindContext, ReturnContext, BaseTypeImpl, BoundVariable, AccessType, PrimType, BoundVariableRuntime, CallContext, Shape

from kernelfunctions.core.reflection import TYPE_OVERRIDES, SlangProgramLayout, SlangType, TypeReflection
from kernelfunctions.types import NDBuffer, NDDifferentiableBuffer

from kernelfunctions.backend import ResourceUsage
from kernelfunctions.typeregistry import PYTHON_TYPES


def _calc_broadcast(context: CallContext, binding: BoundVariableRuntime):
    broadcast = []
    transform = cast(Shape, binding.transform)
    for i in range(len(transform)):
        csidx = transform[i]
        broadcast.append(context.call_shape[csidx] != binding.shape[i])
    broadcast.extend([False]*(len(binding.shape) - len(broadcast)))
    return broadcast


class NDBufferType(SlangType):
    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):
        args = program.get_resolved_generic_args(refl)
        assert args is not None
        assert len(args) == 2
        assert isinstance(args[0], SlangType)
        assert isinstance(args[1], int)
        super().__init__(program, refl,
                         element_type=args[0], local_shape=Shape((-1,)*args[1]))
        self._writable = self.type_reflection.full_name.startswith('RW')

    @property
    def writable(self) -> bool:
        return self._writable


TYPE_OVERRIDES["NDBuffer"] = NDBufferType
TYPE_OVERRIDES["RWNDBuffer"] = NDBufferType


class BaseNDBufferMarshall(BaseTypeImpl):
    def __init__(self, layout: SlangProgramLayout, element_type: SlangType, dims: int, writable: bool):
        super().__init__(layout)

        self.dims = dims
        self.writable = writable

        prefix = "RW" if self.writable else ""

        # Note: find by name handles the fact that element type may not be from the same program layout
        slet = layout.find_type_by_name(element_type.full_name)
        assert slet is not None
        self.slang_element_type = slet

        slt = layout.find_type_by_name(
            f"{prefix}NDBuffer<{self.slang_element_type.full_name},{self.dims}>")
        assert slt is not None
        self.slang_type = slt

    def reduce_type(self, context: BindContext, dimensions: int):
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

    def resolve_type(self, context: BindContext, bound_type: 'SlangType'):

        if isinstance(bound_type, NDBufferType):
            # If the bound type is an NDBuffer, verify properties match then just use it
            if bound_type.writable and not self.writable:
                raise ValueError(
                    "Attempted to bind a writable buffer to a read-only buffer")
            if bound_type.element_type != self.slang_element_type:
                raise ValueError(
                    "Attempted to bind a buffer with a different element type")
            return bound_type

        # if implicit element casts enabled, allow conversion from type to element type
        if context.options['implicit_element_casts']:
            if self.slang_element_type == bound_type:
                return bound_type

        # if implicit tensor casts enabled, allow conversion from vector/matrix to element type
        if context.options['implicit_tensor_casts']:
            if bound_type.full_name.startswith('vector<') and self.slang_element_type == bound_type.element_type:
                return bound_type
            elif bound_type.full_name.startswith('matrix<') and self.slang_element_type == bound_type.element_type:
                return bound_type

        # Default to just casting to itself (i.e. no implicit cast)
        return self.slang_type

    def resolve_dimensionality(self, context: BindContext, binding: BoundVariable, vector_target_type: 'SlangType'):
        return self.dims + len(self.slang_element_type.shape) - len(vector_target_type.shape)

    def get_shape(self, value: Optional[NDBuffer] = None) -> Shape:
        if value is not None:
            return value.shape+self.slang_element_type.shape
        else:
            return Shape((-1,)*self.dims)+self.slang_element_type.shape

    @property
    def is_writable(self) -> bool:
        return self.writable


class NDBufferMarshall(BaseNDBufferMarshall):

    def __init__(self, layout: SlangProgramLayout, element_type: SlangType, dims: int, writable: bool):
        super().__init__(layout, element_type, dims, writable)

    @property
    def has_derivative(self) -> bool:
        return False

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: 'BoundVariable'):
        access = binding.access
        name = binding.variable_name
        assert access[0] != AccessType.none
        assert access[1] == AccessType.none
        if isinstance(binding.vector_type, NDBufferType):
            # If passing to NDBuffer, just use the NDBuffer type
            assert access[0] == AccessType.read
            assert isinstance(binding.vector_type, NDBufferType)
            cgb.type_alias(f"_t_{name}", binding.vector_type.full_name)
        else:
            # If broadcasting to an element, use the type of this buffer for code gen
            if access[0] == AccessType.read:
                cgb.type_alias(
                    f"_t_{name}", f"NDBuffer<{self.slang_element_type.full_name},{self.dims}>")
            else:
                cgb.type_alias(
                    f"_t_{name}", f"RWNDBuffer<{self.slang_element_type.full_name},{self.dims}>")

    def create_calldata(self, context: CallContext, binding: 'BoundVariableRuntime', data: NDBuffer) -> Any:
        if isinstance(binding.vector_type, NDBufferType):
            return {
                'buffer': data.buffer,
                'strides': data.strides
            }
        else:
            broadcast = _calc_broadcast(context, binding)
            return {
                'buffer': data.buffer,
                'strides': [data.strides[i] if not broadcast[i] else 0 for i in range(len(data.strides))]
            }

    def create_output(self, context: CallContext, binding: BoundVariableRuntime) -> Any:
        et = slang_type_to_return_type(self.slang_element_type)
        return NDBuffer(context.device, et, shape=context.call_shape, usage=ResourceUsage.shader_resource | ResourceUsage.unordered_access)

    def read_output(self, context: CallContext, binding: BoundVariableRuntime, data: NDDifferentiableBuffer) -> Any:
        return data


def create_vr_type_for_value(layout: SlangProgramLayout, value: Any):
    if isinstance(value, NDBuffer):
        return NDBufferMarshall(layout, value.element_type,
                                len(value.shape),
                                (value.usage & ResourceUsage.unordered_access) != 0)
    elif isinstance(value, ReturnContext):
        return NDBufferMarshall(layout, value.slang_type,
                                value.bind_context.call_dimensionality,
                                True)
    else:
        raise ValueError(
            f"Unexpected type {type(value)} attempting to create NDBuffer marshall")


PYTHON_TYPES[NDBuffer] = create_vr_type_for_value


def generate_differential_buffer(name: str, context: str, primal_storage: str, deriv_storage: str, primal_target: str, deriv_target: Optional[str]):
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

    def __init__(self, layout: SlangProgramLayout, element_type: SlangType, dims: int, writable: bool):
        super().__init__(layout, element_type, dims, writable)

        if not element_type.differentiable:
            raise ValueError(f"Elements of differentiable buffer must be differentiable.")

    @property
    def has_derivative(self) -> bool:
        return True

    # Call data can only be read access to primal, and simply declares it as a variable
    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: 'BoundVariable'):
        access = binding.access
        name = binding.variable_name

        if isinstance(binding.vector_type, NDBufferType):
            # If passing to NDBuffer, just use the NDBuffer type
            assert access[0] == AccessType.read
            assert isinstance(binding.vector_type, NDBufferType)
            cgb.type_alias(f"_t_{name}", binding.vector_type.full_name)
        else:
            # If broadcasting to an element, use full diff pair logic
            prim_el = self.slang_element_type.full_name
            deriv_el = prim_el + ".Differential"
            dim = self.dims

            if access[0] == AccessType.none:
                primal_storage = f'NoneType'
            elif access[0] == AccessType.read:
                primal_storage = f"NDBuffer<{prim_el},{dim}>"
            else:
                primal_storage = f"RWNDBuffer<{prim_el},{dim}>"

            if access[1] == AccessType.none:
                deriv_storage = f'NoneType'
            elif access[1] == AccessType.read:
                deriv_storage = f"NDBuffer<{deriv_el},{dim}>"
            else:
                deriv_storage = f"RWNDBuffer<{deriv_el},{dim}>"

            assert binding.vector_type is not None
            primal_target = binding.vector_type.full_name
            deriv_target = binding.vector_type.full_name + ".Differential"

            slang_context = f"ContextND<{binding.call_dimensionality}>"

            cgb.append_code_indented(generate_differential_buffer(name, slang_context, primal_storage,
                                                                  deriv_storage, primal_target, deriv_target))

    def create_calldata(self, context: CallContext, binding: 'BoundVariableRuntime', data: NDDifferentiableBuffer) -> Any:
        if isinstance(binding.vector_type, NDBufferType):
            return {
                'buffer': data.buffer,
                'strides': data.strides
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
                    value = ndbuffer.buffer if prim == PrimType.primal else ndbuffer.buffer
                    res[prim_name] = {
                        'buffer': value,
                        'strides': [data.strides[i] if not broadcast[i] else 0 for i in range(len(data.strides))]
                    }
            return res

    def create_output(self, context: CallContext, binding: BoundVariableRuntime) -> Any:
        et = slang_type_to_return_type(self.slang_element_type)
        return NDDifferentiableBuffer(context.device, et,
                                      shape=context.call_shape,
                                      requires_grad=True,
                                      usage=ResourceUsage.shader_resource | ResourceUsage.unordered_access)

    def read_output(self, context: CallContext, binding: BoundVariableRuntime, data: NDDifferentiableBuffer) -> Any:
        return data


def create_gradvr_type_for_value(layout: SlangProgramLayout, value: Any):
    if isinstance(value, NDDifferentiableBuffer):
        return NDDifferentiableBufferMarshall(layout, value.element_type,
                                              len(value.shape),
                                              (value.usage & ResourceUsage.unordered_access) != 0)
    elif isinstance(value, ReturnContext):
        return NDDifferentiableBufferMarshall(layout, value.slang_type,
                                              value.bind_context.call_dimensionality,
                                              True)
    else:
        raise ValueError(
            f"Unexpected type {type(value)} attempting to create NDDifferentiableBuffer marshall")


PYTHON_TYPES[NDDifferentiableBuffer] = create_gradvr_type_for_value
