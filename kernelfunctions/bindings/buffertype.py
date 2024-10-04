

from typing import Any, Optional, cast

from sgl import TypeReflection

from kernelfunctions.bindings.diffpairtype import generate_differential_pair
from kernelfunctions.core import CodeGenBlock, BaseType, BaseTypeImpl, BoundVariable, AccessType, PrimType, BoundVariableRuntime, CallContext, Shape

from kernelfunctions.types import NDBuffer, NDDifferentiableBuffer

from kernelfunctions.backend import ResourceUsage
from kernelfunctions.typeregistry import PYTHON_TYPES, SLANG_STRUCT_TYPES_BY_NAME, get_or_create_type
from kernelfunctions.utils import parse_generic_signature


def _calc_broadcast(context: CallContext, binding: BoundVariableRuntime):
    broadcast = []
    transform = cast(Shape, binding.transform)
    full_cs = context.call_shape + binding.slang_shape
    for i in range(len(transform)):
        csidx = transform[i]
        broadcast.append(full_cs[csidx] != binding.shape[i])
    return broadcast


class NDBufferType(BaseTypeImpl):

    def __init__(self, element_type: BaseType, dims: int, writable: bool):
        super().__init__()
        self.element_type = element_type
        self.dims = dims
        self.writable = writable

        if not self.writable:
            self.name = f"NDBuffer<{self.element_type.name},{self.dims}>"
        else:
            self.name = f"RWNDBuffer<{self.element_type.name},{self.dims}>"

    # Values don't store a derivative - they're just a value
    @property
    def has_derivative(self) -> bool:
        return False

    @property
    def is_writable(self) -> bool:
        return self.writable

    # Call data can only be read access to primal, and simply declares it as a variable
    def gen_calldata(self, cgb: CodeGenBlock, binding: 'BoundVariable'):
        access = binding.access
        name = binding.variable_name
        assert self.element_type is not None
        assert access[0] != AccessType.none
        assert access[1] == AccessType.none
        if access[0] == AccessType.read:
            cgb.type_alias(
                f"_{name}", f"NDBuffer<{self.element_type.name},{self.dims}>")
        else:
            cgb.type_alias(
                f"_{name}", f"RWNDBuffer<{self.element_type.name},{self.dims}>")

    # Call data just returns the primal

    def create_calldata(self, context: CallContext, binding: 'BoundVariableRuntime', data: NDBuffer) -> Any:
        broadcast = _calc_broadcast(context, binding)
        return {
            'buffer': data.buffer,
            'strides': [data.strides[i] if not broadcast[i] else 0 for i in range(len(data.strides))],
            'transform': binding.transform.as_tuple()[0:self.dims]
        }

    def get_container_shape(self, value: Optional[NDDifferentiableBuffer] = None) -> Shape:
        if value is not None:
            return value.shape
        else:
            return Shape((-1,)*self.dims)

    @property
    def differentiable(self):
        return self.element_type.differentiable

    @property
    def derivative(self):
        et = self.element_type
        if et is not None:
            return NDBufferType(et, self.dims, self.writable)
        else:
            return None

    def create_output(self, context: CallContext) -> Any:
        return NDBuffer(context.device, self.element_type.python_return_value_type, shape=context.call_shape, usage=ResourceUsage.shader_resource | ResourceUsage.unordered_access)

    def read_output(self, context: CallContext, data: NDDifferentiableBuffer) -> Any:
        return data


def create_vr_type_for_value(value: NDBuffer):
    assert isinstance(value, NDBuffer)
    return NDBufferType(get_or_create_type(value.element_type), len(value.shape), (value.usage & ResourceUsage.unordered_access) != 0)


def create_vr_type_for_slang(value: TypeReflection):
    assert isinstance(value, TypeReflection)
    name, args = parse_generic_signature(value.full_name)
    return NDBufferType(get_or_create_type(args[0]), int(args[1]), name.startswith("RW"))


PYTHON_TYPES[NDBuffer] = create_vr_type_for_value
SLANG_STRUCT_TYPES_BY_NAME["NDBuffer"] = create_vr_type_for_slang


class NDDifferentiableBufferType(BaseTypeImpl):

    def __init__(self, element_type: BaseType, dims: int, writable: bool):
        super().__init__()
        self.element_type = element_type
        self.dims = dims
        self.writable = writable

        if not self.writable:
            self.name = f"NDBuffer<{self.element_type.name},{self.dims}>"
        else:
            self.name = f"RWNDBuffer<{self.element_type.name},{self.dims}>"

    # Values don't store a derivative - they're just a value
    @property
    def has_derivative(self) -> bool:
        return True

    @property
    def is_writable(self) -> bool:
        return self.writable

    # Call data can only be read access to primal, and simply declares it as a variable
    def gen_calldata(self, cgb: CodeGenBlock, binding: 'BoundVariable'):
        access = binding.access
        name = binding.variable_name

        prim_el = binding.python.primal_element_name
        deriv_el = binding.python.derivative_element_name
        if deriv_el is None:
            deriv_el = prim_el
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

        primal_target = binding.slang.primal_type_name
        deriv_target = binding.slang.derivative_type_name

        cgb.append_code(generate_differential_pair(name, primal_storage,
                        deriv_storage, primal_target, deriv_target))

    # Call data just returns the primal

    def create_calldata(self, context: CallContext, binding: 'BoundVariableRuntime', data: NDDifferentiableBuffer) -> Any:
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
                    'strides': [data.strides[i] if not broadcast[i] else 0 for i in range(len(data.strides))],
                    'transform': binding.transform.as_tuple()[0:self.dims]
                }
        return res

    def get_container_shape(self, value: Optional[NDDifferentiableBuffer] = None) -> Shape:
        if value is not None:
            return value.shape
        else:
            return Shape((-1,)*self.dims)

    @property
    def differentiable(self):
        return self.element_type.differentiable

    @property
    def derivative(self):
        et = self.element_type.derivative
        if et is not None:
            return NDDifferentiableBufferType(et, self.dims, self.writable)
        else:
            return None

    def create_output(self, context: CallContext) -> Any:
        return NDDifferentiableBuffer(context.device, self.element_type.python_return_value_type,
                                      shape=context.call_shape,
                                      requires_grad=True,
                                      usage=ResourceUsage.shader_resource | ResourceUsage.unordered_access)

    def read_output(self, context: CallContext, data: NDDifferentiableBuffer) -> Any:
        return data


def create_gradvr_type_for_value(value: Any):
    assert isinstance(value, NDDifferentiableBuffer)
    return NDDifferentiableBufferType(get_or_create_type(value.element_type), len(value.shape), (value.usage & ResourceUsage.unordered_access) != 0)


PYTHON_TYPES[NDDifferentiableBuffer] = create_gradvr_type_for_value
