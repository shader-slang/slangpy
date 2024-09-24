

from typing import Any, Optional, Sequence

from sgl import TypeReflection

from kernelfunctions.bindings.diffpairtype import generate_differential_pair
from kernelfunctions.core import CodeGenBlock, BaseType, BaseTypeImpl, BaseVariable, AccessType, PrimType

from kernelfunctions.types import NDBuffer, NDDifferentiableBuffer

from kernelfunctions.backend import Device, ResourceUsage
from kernelfunctions.typeregistry import PYTHON_TYPES, SLANG_STRUCT_TYPES_BY_NAME, get_or_create_type
from kernelfunctions.utils import parse_generic_signature


class NDBufferType(BaseTypeImpl):

    def __init__(self, element_type: BaseType, dims: int, writable: bool):
        super().__init__()
        self.el_type = element_type
        self.dims = dims
        self.writable = writable

    # Values don't store a derivative - they're just a value
    def has_derivative(self, value: Any = None) -> bool:
        return False

    def is_writable(self, value: Optional[NDBuffer] = None) -> bool:
        return self.writable

    # Call data can only be read access to primal, and simply declares it as a variable
    def gen_calldata(self, cgb: CodeGenBlock, input_value: 'BaseVariable', name: str, transform: list[Optional[int]], access: tuple[AccessType, AccessType]):
        assert access[0] != AccessType.none
        assert access[1] == AccessType.none
        if access[0] == AccessType.read:
            cgb.type_alias(
                f"_{name}", f"NDBuffer<{input_value.primal_element_name},{self.dims}>")
        else:
            cgb.type_alias(
                f"_{name}", f"RWNDBuffer<{input_value.primal_element_name},{self.dims}>")

    # Call data just returns the primal

    def create_calldata(self, device: Device, input_value: 'BaseVariable', access: tuple[AccessType, AccessType], data: NDBuffer) -> Any:
        assert access[0] != AccessType.none
        assert access[1] == AccessType.none
        assert input_value.binding is not None
        return {
            'buffer': data.buffer,
            'strides': list(data.strides),
            'transform': input_value.binding.transform[0:self.dims]
        }

    # Read back from call data does nothing
    def read_calldata(self, device: Device, input_value: 'BaseVariable', access: tuple[AccessType, AccessType], data: NDBuffer, result: Any) -> None:
        pass

    def name(self, value: Any = None) -> str:
        if not self.writable:
            return f"NDBuffer<{self.el_type.name()},{self.dims}>"
        else:
            return f"RWNDBuffer<{self.el_type.name()},{self.dims}>"

    def element_type(self, value: Optional[NDBuffer] = None):
        return self.el_type

    def container_shape(self, value: Optional[NDDifferentiableBuffer] = None):
        if value is not None:
            assert len(value.shape) == self.dims
            return value.shape
        else:
            return [None]*self.dims

    def differentiable(self, value: Optional[NDBuffer] = None):
        return self.el_type.differentiable()

    def differentiate(self, value: Optional[NDBuffer] = None):
        et = self.el_type.differentiate()
        if et is not None:
            return NDBufferType(et, self.dims, self.writable)
        else:
            return None

    def create_output(self, device: Device, call_shape: Sequence[int]) -> Any:
        return NDBuffer(device, self.el_type.python_return_value_type(), shape=tuple(call_shape), usage=ResourceUsage.shader_resource | ResourceUsage.unordered_access)

    def read_output(self, device: Device, data: NDDifferentiableBuffer) -> Any:
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
        self.el_type = element_type
        self.dims = dims
        self.writable = writable

    # Values don't store a derivative - they're just a value
    def has_derivative(self, value: Any = None) -> bool:
        return True

    def is_writable(self, value: Any = None) -> bool:
        return self.writable

    # Call data can only be read access to primal, and simply declares it as a variable
    def gen_calldata(self, cgb: CodeGenBlock, input_value: 'BaseVariable', name: str, transform: list[Optional[int]], access: tuple[AccessType, AccessType]):
        prim_el = input_value.primal_element_name
        deriv_el = input_value.derivative_element_name
        if deriv_el is None:
            deriv_el = prim_el
        dim = self.dims

        binding = input_value.binding

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

        assert binding is not None
        primal_target = binding.slang.primal_type_name
        deriv_target = binding.slang.derivative_type_name

        cgb.append_code(generate_differential_pair(name, primal_storage,
                        deriv_storage, primal_target, deriv_target))

    # Call data just returns the primal

    def create_calldata(self, device: Device, input_value: 'BaseVariable', access: tuple[AccessType, AccessType], data: NDDifferentiableBuffer) -> Any:
        assert input_value.binding is not None
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
                    'strides': list(data.strides),
                    'transform': input_value.binding.transform[0:self.dims]
                }
        return res

    # Read back from call data does nothing
    def read_calldata(self, device: Device, input_value: 'BaseVariable', access: tuple[AccessType, AccessType], data: NDDifferentiableBuffer, result: Any) -> None:
        pass

    def name(self, value: Optional[NDDifferentiableBuffer] = None) -> str:
        if not self.writable:
            return f"NDBuffer<{self.el_type.name()},{self.dims}>"
        else:
            return f"RWNDBuffer<{self.el_type.name()},{self.dims}>"

    def element_type(self, value: Optional[NDDifferentiableBuffer] = None):
        return self.el_type

    def container_shape(self, value: Optional[NDDifferentiableBuffer] = None):
        if value is not None:
            assert len(value.shape) == self.dims
            return value.shape
        else:
            return [None]*self.dims

    def differentiable(self, value: Optional[NDBuffer] = None):
        return self.el_type.differentiable()

    def differentiate(self, value: Optional[NDBuffer] = None):
        et = self.el_type.differentiate()
        if et is not None:
            return NDDifferentiableBufferType(et, self.dims, self.writable)
        else:
            return None

    def create_output(self, device: Device, call_shape: Sequence[int]) -> Any:
        return NDDifferentiableBuffer(device, self.el_type.python_return_value_type(),
                                      shape=tuple(call_shape),
                                      requires_grad=True,
                                      usage=ResourceUsage.shader_resource | ResourceUsage.unordered_access)

    def read_output(self, device: Device, data: NDDifferentiableBuffer) -> Any:
        return data


def create_gradvr_type_for_value(value: Any):
    assert isinstance(value, NDDifferentiableBuffer)
    return NDDifferentiableBufferType(get_or_create_type(value.element_type), len(value.shape), (value.usage & ResourceUsage.unordered_access) != 0)


PYTHON_TYPES[NDDifferentiableBuffer] = create_gradvr_type_for_value
