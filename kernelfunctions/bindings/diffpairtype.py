

from typing import Any, Optional, Sequence
import numpy as np

from kernelfunctions.core import CodeGenBlock, BaseType, BaseTypeImpl, BoundVariable, AccessType, PrimType

from kernelfunctions.types import DiffPair

from kernelfunctions.backend import Buffer, Device, ResourceUsage
from kernelfunctions.typeregistry import PYTHON_TYPES, get_or_create_type


def generate_differential_pair(name: str, primal_storage: str, deriv_storage: str, primal_target: str, deriv_target: Optional[str]):
    assert primal_storage
    assert deriv_storage
    assert primal_target
    if deriv_target is None:
        deriv_target = primal_target

    DIFF_PAIR_CODE = f"""
struct _{name}
{{
    {primal_storage} primal;
    {deriv_storage} derivative;
    void load_primal(IContext context, out {primal_target} value) {{ primal.load_primal(context, value); }}
    void store_primal(IContext context, in {primal_target} value) {{ primal.store_primal(context, value); }}
    void load_derivative(IContext context, out {deriv_target} value) {{ derivative.load_primal(context, value); }}
    void store_derivative(IContext context, in {deriv_target} value) {{ derivative.store_primal(context, value); }}
}}
"""
    return DIFF_PAIR_CODE


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
    @property
    def is_writable(self) -> bool:
        return True

    # Call data can only be read access to primal, and simply declares it as a variable
    def gen_calldata(self, cgb: CodeGenBlock, input_value: 'BoundVariable', name: str, transform: list[Optional[int]], access: tuple[AccessType, AccessType]):
        prim_el = input_value.python.primal_element_name
        deriv_el = input_value.python.derivative_element_name
        if deriv_el is None:
            deriv_el = prim_el

        if access[0] == AccessType.none:
            primal_storage = f'NoneType'
        elif access[0] == AccessType.read:
            primal_storage = f"ValueType<{prim_el}>"
        else:
            primal_storage = f"RWValueRef<{prim_el}>"

        if access[1] == AccessType.none:
            deriv_storage = f'NoneType'
        elif access[1] == AccessType.read:
            deriv_storage = f"ValueType<{deriv_el}>"
        else:
            deriv_storage = f"RWValueRef<{deriv_el}>"

        primal_target = input_value.slang.primal_type_name
        deriv_target = input_value.slang.derivative_type_name

        cgb.append_code(generate_differential_pair(name, primal_storage,
                        deriv_storage, primal_target, deriv_target))

    def get_type(self, prim: PrimType):
        return self.primal_type if prim == PrimType.primal else self.derivative_type

    # Call data just returns the primal
    def create_calldata(self, device: Device, input_value: 'BoundVariable', access: tuple[AccessType, AccessType], broadcast: list[bool], data: DiffPair) -> Any:
        res = {}

        for prim in PrimType:
            prim_name = prim.name
            prim_access = access[prim.value]
            prim_data = data.get(prim)
            prim_type = self.get_type(prim)
            if prim_access in [AccessType.write, AccessType.readwrite]:
                assert prim_type is not None
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
    def read_calldata(self, device: Device, input_value: 'BoundVariable', access: tuple[AccessType, AccessType], data: DiffPair, result: Any) -> None:
        for prim in PrimType:
            prim_name = prim.name
            prim_access = access[prim.value]
            prim_type = self.get_type(prim)
            if prim_access in [AccessType.write, AccessType.readwrite]:
                assert isinstance(result[prim_name]['value'], Buffer)
                assert prim_type is not None
                npdata = result[prim_name]['value'].to_numpy()
                data.set(prim, prim_type.from_numpy(npdata))

    @property
    def name(self) -> str:
        return self.primal_type.name

    @property
    def element_type(self):
        return self.primal_type.element_type

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
