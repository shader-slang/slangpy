

from typing import Any, Optional, Sequence

from sgl import Device, ResourceUsage
from kernelfunctions.codegen import CodeGenBlock
from kernelfunctions.typeregistry import PYTHON_TYPES, get_or_create_type
from kernelfunctions.types.basetype import BaseType
from kernelfunctions.types.basetypeimpl import BaseTypeImpl
from kernelfunctions.types.basevariable import BaseVariable
from kernelfunctions.types.enums import AccessType, PrimType
from kernelfunctions.types.buffer import NDBuffer, NDDifferentiableBuffer

TYPES = r"""
int _idx<let N: int>(int[N] index, int[N] stride) {
    int idx = 0;
    for (int i = 0; i < N; i++) { idx += index[i] * stride[i]; }
    return idx;
}
struct TensorBuffer<T, let N : int> {
    RWStructuredBuffer<T> buffer;
    int[N] strides;
    T get(int[N] index) { return buffer[_idx(index, strides)]; }
    __subscript(int[N] index)->T { get { return get(index); } }
}
struct RWTensorBuffer<T, let N : int> {
    RWStructuredBuffer<T> buffer;
    int[N] strides;
    T get(int[N] index) { return buffer[_idx(index, strides)]; }
    void set(int[N] index, T value) { buffer[_idx(index, strides)] = value; }
    __subscript(int[N] index)->T { get { return get(index); } set { set(index, newValue); } }
}
"""


def _transform_to_subscript(transform: list[Optional[int]]):
    """
    Generates the subscript to be passed into the [] operator when loading or storing
    from the buffer.
    """
    vals = ",".join(
        ("0" if x is None else f"context.call_id[{x}]") for x in transform)
    return f"[{{{vals}}}]"


class NDBufferType(BaseTypeImpl):

    def __init__(self, element_type: BaseType):
        super().__init__()
        self.el_type = element_type

    # Values don't store a derivative - they're just a value
    def has_derivative(self, value: Any = None) -> bool:
        return False

    # Refs can be written to!
    def is_writable(self, value: Optional[NDBuffer] = None) -> bool:
        if value is not None:
            return (value.usage & ResourceUsage.unordered_access) != 0
        else:
            return True  # to be allocated later for write!

    # Call data can only be read access to primal, and simply declares it as a variable
    def gen_calldata(self, cgb: CodeGenBlock, input_value: 'BaseVariable', name: str, transform: list[Optional[int]], access: tuple[AccessType, AccessType]):
        assert access[0] != AccessType.none
        assert access[1] == AccessType.none
        if access[0] == AccessType.read:
            cgb.type_alias(
                f"_{name}", f"TensorBuffer<{input_value.primal_element_name},{len(transform)}>")
        else:
            cgb.type_alias(
                f"_{name}", f"RWTensorBuffer<{input_value.primal_element_name},{len(transform)}>")

    # Call data just returns the primal

    def create_calldata(self, device: Device, input_value: 'BaseVariable', access: tuple[AccessType, AccessType], data: NDBuffer) -> Any:
        assert access[0] != AccessType.none
        assert access[1] == AccessType.none
        return {
            'buffer': data.buffer,
            'strides': list(data.strides)
        }

    # Read back from call data does nothing
    def read_calldata(self, device: Device, input_value: 'BaseVariable', access: tuple[AccessType, AccessType], data: NDBuffer, result: Any) -> None:
        pass

    def name(self, value: Any = None) -> str:
        if value is not None:
            if self.is_writable(value):
                return f"TensorBuffer<{self.el_type.name()}>"
            else:
                return f"RWTensorBuffer<{self.el_type.name()}>"
        else:
            return "UnknownBufferName"

    def element_type(self, value: Optional[NDBuffer] = None):
        return self.el_type

    def container_shape(self, value: Optional[NDBuffer] = None):
        if value is not None:
            return value.shape
        else:
            return None

    def shape(self, value: Any = None):
        if value is not None:
            return super().shape(value)
        else:
            return None

    def differentiable(self, value: Optional[NDBuffer] = None):
        return self.el_type.differentiable()

    def differentiate(self, value: Optional[NDBuffer] = None):
        et = self.el_type.differentiate()
        if et is not None:
            return NDBufferType(et)
        else:
            return None

    def create_output(self, device: Device, call_shape: Sequence[int]) -> Any:
        return NDBuffer(device, self.el_type.python_return_value_type(), shape=tuple(call_shape), usage=ResourceUsage.shader_resource | ResourceUsage.unordered_access)

    def read_output(self, device: Device, data: NDDifferentiableBuffer) -> Any:
        return data


def create_vr_type_for_value(value: Any):
    assert isinstance(value, NDBuffer)
    return NDBufferType(get_or_create_type(value.element_type))


PYTHON_TYPES[NDBuffer] = create_vr_type_for_value


class NDDifferentiableBufferType(BaseTypeImpl):

    def __init__(self, element_type: BaseType):
        super().__init__()
        self.el_type = element_type

    # Values don't store a derivative - they're just a value
    def has_derivative(self, value: Any = None) -> bool:
        return True

    # Refs can be written to!
    def is_writable(self, value: Any = None) -> bool:
        if value is not None:
            return (value.usage & ResourceUsage.unordered_access) != 0
        else:
            return True  # to be allocated later for write!

    # Call data can only be read access to primal, and simply declares it as a variable
    def gen_calldata(self, cgb: CodeGenBlock, input_value: 'BaseVariable', name: str, transform: list[Optional[int]], access: tuple[AccessType, AccessType]):
        prim_el = input_value.primal_element_name
        deriv_el = input_value.derivative_element_name
        if deriv_el is None:
            deriv_el = prim_el
        dim = len(transform)

        if access[0] == AccessType.none:
            primal_storage = f'NoneType<{prim_el}>'
        elif access[0] == AccessType.read:
            primal_storage = f"TensorBuffer<{prim_el},{dim}>"
        else:
            primal_storage = f"RWTensorBuffer<{prim_el},{dim}>"

        if access[1] == AccessType.none:
            deriv_storage = f'NoneType<{deriv_el}>'
        elif access[1] == AccessType.read:
            deriv_storage = f"TensorBuffer<{deriv_el},{dim}>"
        else:
            deriv_storage = f"RWTensorBuffer<{deriv_el},{dim}>"

        tn = f"BaseDiffPair<{prim_el},{deriv_el},{primal_storage},{deriv_storage}>"
        cgb.type_alias(f"_{name}", tn)

    # Call data just returns the primal

    def create_calldata(self, device: Device, input_value: 'BaseVariable', access: tuple[AccessType, AccessType], data: NDDifferentiableBuffer) -> Any:
        res = {}
        for prim in PrimType:
            prim_name = prim.name
            prim_access = access[prim.value]
            if prim_access != AccessType.none:
                value = data.buffer if prim == PrimType.primal else data.grad_buffer
                res[prim_name] = {
                    'buffer': value,
                    'strides': list(data.strides)
                }
        return res

    # Read back from call data does nothing
    def read_calldata(self, device: Device, input_value: 'BaseVariable', access: tuple[AccessType, AccessType], data: NDDifferentiableBuffer, result: Any) -> None:
        pass

    def name(self, value: Optional[NDDifferentiableBuffer] = None) -> str:
        if value is not None:
            if self.is_writable(value):
                return f"TensorBuffer<{self.el_type.name()}>"
            else:
                return f"RWTensorBuffer<{self.el_type.name()}>"
        else:
            return "UnknownBufferName"

    def element_type(self, value: Optional[NDDifferentiableBuffer] = None):
        return self.el_type

    def container_shape(self, value: Optional[NDDifferentiableBuffer] = None):
        if value is not None:
            return value.shape
        else:
            return None

    def shape(self, value: Any = None):
        if value is not None:
            return super().shape(value)
        else:
            return None

    def differentiable(self, value: Optional[NDBuffer] = None):
        return self.el_type.differentiable()

    def differentiate(self, value: Optional[NDBuffer] = None):
        et = self.el_type.differentiate()
        if et is not None:
            return NDDifferentiableBufferType(et)
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
    return NDDifferentiableBufferType(get_or_create_type(value.element_type))


PYTHON_TYPES[NDDifferentiableBuffer] = create_gradvr_type_for_value
