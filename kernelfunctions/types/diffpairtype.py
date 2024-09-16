

from typing import Any, Optional, Sequence
import numpy as np

from sgl import Buffer, Device, ResourceUsage
from kernelfunctions.codegen import CodeGenBlock
from kernelfunctions.typeregistry import PYTHON_TYPES, get_or_create_type
from kernelfunctions.types.basetype import BaseType
from kernelfunctions.types.basetypeimpl import BaseTypeImpl
from kernelfunctions.types.basevalue import BaseVariable
from kernelfunctions.types.enums import AccessType, PrimType
from kernelfunctions.types.diffpair import DiffPair


class DiffPairType(BaseTypeImpl):

    def __init__(self, primal_type: BaseType, derivative_type: Optional[BaseType] = None):
        super().__init__()
        self.primal_type = primal_type
        self.derivative_type = derivative_type

    # Values don't store a derivative - they're just a value
    def has_derivative(self, value: Optional[DiffPair] = None) -> bool:
        assert value is not None
        return value.needs_grad and self.derivative_type != None

    # Refs can be written to!
    def is_writable(self, value: Any = None) -> bool:
        return True

    # Call data can only be read access to primal, and simply declares it as a variable
    def gen_calldata(self, cgb: CodeGenBlock, input_value: 'BaseVariable', name: str, transform: list[Optional[int]], access: tuple[AccessType, AccessType]):
        prim_el = input_value.primal_element_name
        deriv_el = input_value.derivative_element_name
        if deriv_el is None:
            deriv_el = prim_el

        if access[0] == AccessType.none:
            primal_storage = f'NoneType<{prim_el}>'
        elif access[0] == AccessType.read:
            primal_storage = f"ValueType<{prim_el}>"
        else:
            primal_storage = f"RWValueRef<{prim_el}>"

        if access[1] == AccessType.none:
            deriv_storage = f'NoneType<{deriv_el}>'
        elif access[1] == AccessType.read:
            deriv_storage = f"ValueType<{deriv_el}>"
        else:
            deriv_storage = f"RWValueRef<{deriv_el}>"

        tn = f"BaseDiffPair<{prim_el},{deriv_el},{primal_storage},{deriv_storage}>"
        cgb.type_alias(f"_{name}", tn)

    def get_type(self, prim: PrimType):
        return self.primal_type if prim == PrimType.primal else self.derivative_type

    # Call data just returns the primal
    def create_calldata(self, device: Device, input_value: 'BaseVariable', access: tuple[AccessType, AccessType], data: DiffPair) -> Any:
        res = {}

        for prim in PrimType:
            prim_name = prim.name
            prim_access = access[prim.value]
            prim_data = data.get(prim)
            prim_type = self.get_type(prim)
            if prim_access in [AccessType.write, AccessType.readwrite]:
                npdata = prim_type.to_numpy(prim_data).view(dtype=np.uint8)
                res[prim_name] = {
                    'value': device.create_buffer(
                        element_count=1,
                        struct_size=npdata.size,
                        data=npdata,
                        usage=ResourceUsage.shader_resource | ResourceUsage.unordered_access)}
            elif prim_access == AccessType.read:
                res[prim_name] = {'value': prim_data}

        return res

    # Read back from call data does nothing
    def read_calldata(self, device: Device, input_value: 'BaseVariable', access: tuple[AccessType, AccessType], data: DiffPair, result: Any) -> None:
        for prim in PrimType:
            prim_name = prim.name
            prim_access = access[prim.value]
            prim_type = self.get_type(prim)
            if prim_access in [AccessType.write, AccessType.readwrite]:
                assert isinstance(result[prim_name]['value'], Buffer)
                npdata = result[prim_name]['value'].to_numpy()
                data.set(prim, prim_type.from_numpy(npdata))

    def name(self, value: Any = None) -> str:
        return self.primal_type.name()

    def element_type(self, value: Optional[DiffPair] = None):
        return self.primal_type.element_type()

    def shape(self, value: Optional[DiffPair] = None):
        return self.primal_type.shape()

    def differentiable(self, value: Optional[DiffPair] = None):
        return self.primal_type.differentiable()

    def differentiate(self, value: Optional[DiffPair] = None):
        return self.primal_type.differentiate()

    def create_output(self, device: Device, call_shape: Sequence[int]) -> Any:
        return DiffPair(None, None)

    def read_output(self, device: Device, data: DiffPair) -> Any:
        return data


def create_vr_type_for_value(value: Any):
    assert isinstance(value, DiffPair)
    return DiffPairType(get_or_create_type(type(value.primal)), get_or_create_type(type(value.grad)))


PYTHON_TYPES[DiffPair] = create_vr_type_for_value
