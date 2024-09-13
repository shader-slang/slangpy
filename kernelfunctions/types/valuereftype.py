

from typing import Any, Optional, Sequence
import numpy as np

from sgl import Buffer, Device, ResourceUsage
from kernelfunctions.codegen import CodeGenBlock
from kernelfunctions.typeregistry import PYTHON_TYPES, get_or_create_type
from kernelfunctions.types.basetype import BaseType
from kernelfunctions.types.basetypeimpl import BaseTypeImpl
from kernelfunctions.types.basevalue import BaseValue
from kernelfunctions.types.enums import AccessType, PrimType
from kernelfunctions.types.valueref import ValueRef


class ValueRefType(BaseTypeImpl):

    def __init__(self, value_type: BaseType):
        super().__init__()
        self.value_type = value_type

    # Values don't store a derivative - they're just a value
    def has_derivative(self) -> bool:
        return False

    # Call data can only be read access to primal, and simply declares it as a variable
    def gen_calldata(self, cgb: CodeGenBlock, input_value: 'BaseValue', name: str, access: tuple[AccessType, AccessType]):
        assert access[0] != AccessType.none
        assert access[1] == AccessType.none
        if access == AccessType.read:
            cgb.declare(input_value.primal_type_name, f"{name}_primal")
        else:
            cgb.declare(
                f"RWStructuredBuffer<{input_value.primal_type_name}>", f"{name}_primal")

    # Load should only ever be reading the primal directly from the call data
    def gen_load(self, cgb: CodeGenBlock, input_value: 'BaseValue', from_call_data: str, to_variable: str, transform: list[Optional[int]], prim: PrimType, access: AccessType):
        assert prim == PrimType.primal
        if access == AccessType.read:
            cgb.assign(to_variable, from_call_data)
        else:
            cgb.assign(to_variable, f"{from_call_data}[0]")

    # Never store anything
    def gen_store(self, cgb: CodeGenBlock, input_value: 'BaseValue', from_variable: str, to_call_data: str, transform: list[Optional[int]], prim: PrimType, access: AccessType):
        assert prim == PrimType.primal
        if access in [AccessType.write, AccessType.readwrite]:
            cgb.assign(f"{to_call_data}[0]", from_variable)

    # Call data just returns the primal
    def create_calldata(self, device: Device, input_value: 'BaseValue', access: tuple[AccessType, AccessType], data: ValueRef) -> Any:
        assert access[0] != AccessType.none
        assert access[1] == AccessType.none
        if access[0] == AccessType.read:
            return data.value
        else:
            npdata = self.value_type.to_numpy(data.value).view(dtype=np.uint8)
            return device.create_buffer(element_count=1, struct_size=npdata.size, data=npdata, usage=ResourceUsage.shader_resource | ResourceUsage.unordered_access)

    # Read back from call data does nothing
    def read_calldata(self, device: Device, input_value: 'BaseValue', access: tuple[AccessType, AccessType], data: ValueRef, result: Any) -> None:
        if access[0] in [AccessType.write, AccessType.readwrite]:
            assert isinstance(result, Buffer)
            npdata = result.to_numpy()
            data.value = self.value_type.from_numpy(npdata)

    def name(self) -> str:
        return self.value_type.name()

    def element_type(self, value: Optional[ValueRef] = None):
        return self.value_type.element_type()

    def shape(self, value: Optional[ValueRef] = None):
        return self.value_type.shape()

    def differentiable(self, value: Optional[ValueRef] = None):
        return self.value_type.differentiable()

    def differentiate(self, value: Optional[ValueRef] = None):
        return self.value_type.differentiate()

    def create_output(self, device: Device, call_shape: Sequence[int]) -> Any:
        return ValueRef(None)

    def read_output(self, device: Device, data: ValueRef) -> Any:
        return data.value


def create_vr_type_for_value(value: Any):
    assert isinstance(value, ValueRef)
    return ValueRefType(get_or_create_type(type(value.value)))


PYTHON_TYPES[ValueRef] = create_vr_type_for_value
