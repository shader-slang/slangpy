

from typing import Any, Optional, Sequence
import numpy as np

from kernelfunctions.core import CodeGenBlock, BaseType, BaseTypeImpl, BoundVariable, AccessType

from kernelfunctions.types import ValueRef

from kernelfunctions.backend import Buffer, Device, ResourceUsage
from kernelfunctions.typeregistry import PYTHON_TYPES, get_or_create_type


class ValueRefType(BaseTypeImpl):

    def __init__(self, value_type: BaseType):
        super().__init__()
        self.value_type = value_type

    # Values don't store a derivative - they're just a value
    @property
    def has_derivative(self) -> bool:
        return False

    # Refs can be written to!
    @property
    def is_writable(self) -> bool:
        return True

    # Call data can only be read access to primal, and simply declares it as a variable
    def gen_calldata(self, cgb: CodeGenBlock, binding: 'BoundVariable'):
        access = binding.access
        name = binding.variable_name
        assert access[0] != AccessType.none
        assert access[1] == AccessType.none
        if access[0] == AccessType.read:
            cgb.type_alias(f"_{name}", f"ValueRef<{self.value_type.name}>")
        else:
            cgb.type_alias(
                f"_{name}", f"RWValueRef<{self.value_type.name}>")

    # Call data just returns the primal

    def create_calldata(self, device: Device, binding: 'BoundVariable', broadcast: list[bool], data: ValueRef) -> Any:
        access = binding.access
        assert access[0] != AccessType.none
        assert access[1] == AccessType.none
        if access[0] == AccessType.read:
            return {'value': data.value}
        else:
            npdata = self.value_type.to_numpy(data.value).view(dtype=np.uint8)
            return {
                'value': device.create_buffer(element_count=1, struct_size=npdata.size, data=npdata, usage=ResourceUsage.shader_resource | ResourceUsage.unordered_access)
            }

    # Read back from call data does nothing
    def read_calldata(self, device: Device, binding: 'BoundVariable', data: ValueRef, result: Any) -> None:
        access = binding.access
        if access[0] in [AccessType.write, AccessType.readwrite]:
            assert isinstance(result['value'], Buffer)
            npdata = result['value'].to_numpy()
            data.value = self.value_type.from_numpy(npdata)

    @property
    def name(self) -> str:
        return self.value_type.name

    @property
    def element_type(self):
        return self.value_type.element_type

    def shape(self, value: Optional[ValueRef] = None):
        return self.value_type.shape()

    @property
    def differentiable(self):
        return self.value_type.differentiable

    @property
    def derivative(self):
        return self.value_type.derivative

    def create_output(self, device: Device, call_shape: Sequence[int]) -> Any:
        return ValueRef(None)

    def read_output(self, device: Device, data: ValueRef) -> Any:
        return data.value


def create_vr_type_for_value(value: Any):
    assert isinstance(value, ValueRef)
    return ValueRefType(get_or_create_type(type(value.value)))


PYTHON_TYPES[ValueRef] = create_vr_type_for_value
