

from typing import Any, Optional, Sequence
import numpy as np

from sgl import Buffer, Device, ResourceUsage
from kernelfunctions.codegen import CodeGenBlock
from kernelfunctions.typeregistry import PYTHON_TYPES, get_or_create_type
from kernelfunctions.types.basetype import BaseType
from kernelfunctions.types.basetypeimpl import BaseTypeImpl
from kernelfunctions.types.basevalue import BaseValue
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
    def gen_calldata(self, cgb: CodeGenBlock, input_value: 'BaseValue', name: str, transform: list[Optional[int]], access: tuple[AccessType, AccessType]):
        cgb.begin_struct(f"_{name}_call_data")
        for prim in PrimType:
            prim_name = prim.name
            prim_access = access[prim.value]
            if prim_access == AccessType.none:
                continue
            cgb.type_alias(f"{prim_name}_type", input_value.primal_type_name)
            if prim_access == AccessType.read:
                cgb.declare(f"{prim_name}_type", prim_name)
                cgb.append_line(
                    f"void load_{prim_name}(Context context, out {prim_name}_type value) {{ value = this.{prim_name}; }}")
            else:
                cgb.declare(f"RWStructuredBuffer<{prim_name}_type>", prim_name)
                cgb.append_line(
                    f"void load_{prim_name}(Context context, out {prim_name}_type value) {{ value = this.{prim_name}[0]; }}")
                cgb.append_line(
                    f"void store_{prim_name}(Context context, in {prim_name}_type value) {{ this.{prim_name}[0] = value; }}")
        cgb.end_struct()

    # Load should only ever be reading the primal directly from the call data
    def gen_load_store(self, cgb: CodeGenBlock, input_value: 'BaseValue', name: str, transform: list[Optional[int]],  access: tuple[AccessType, AccessType]):
        cgb.begin_struct(f"_{name}")
        for prim in PrimType:
            prim_name = prim.name
            prim_access = access[prim.value]
            if prim_access == AccessType.none:
                continue
            cgb.type_alias(f"{prim_name}_type", input_value.primal_type_name)
            if prim_access in [AccessType.read, AccessType.readwrite]:
                cgb.append_line(
                    f"static void load_{prim_name}(Context context, out {prim_name}_type value) {{ call_data.{name}.load_{prim_name}(context,value); }}")
            if prim_access in [AccessType.write, AccessType.readwrite]:
                cgb.append_line(
                    f"static void store_{prim_name}(Context context, in {prim_name}_type value) {{ call_data.{name}.store_{prim_name}(context,value); }}")
        cgb.end_struct()

    # Call data just returns the primal
    def create_calldata(self, device: Device, input_value: 'BaseValue', access: tuple[AccessType, AccessType], data: DiffPair) -> Any:
        res = {}
        for prim in PrimType:
            prim_name = prim.name
            prim_access = access[prim.value]
            if prim_access == AccessType.none:
                continue
            value = data.primal if prim == PrimType.primal else data.grad
            if prim_access == AccessType.read:
                if prim == PrimType.primal:
                    res[prim_name] = data.primal
                else:
                    res[prim_name] = data.grad
            else:
                if prim == PrimType.primal:
                    npdata = self.primal_type.to_numpy(value).view(dtype=np.uint8)
                else:
                    npdata = self.derivative_type.to_numpy(
                        value).view(dtype=np.uint8)  # type: ignore
                res[prim_name] = device.create_buffer(
                    element_count=1, struct_size=npdata.size, data=npdata, usage=ResourceUsage.shader_resource | ResourceUsage.unordered_access)
        return res

    # Read back from call data does nothing
    def read_calldata(self, device: Device, input_value: 'BaseValue', access: tuple[AccessType, AccessType], data: DiffPair, result: Any) -> None:
        for prim in PrimType:
            prim_name = prim.name
            prim_access = access[prim.value]
            if prim_access in [AccessType.write, AccessType.readwrite]:
                value = result[prim_name]
                assert isinstance(value, Buffer)
                npdata = value.to_numpy()
                if prim == PrimType.primal:
                    data.primal = self.primal_type.from_numpy(npdata)
                else:
                    data.grad = self.derivative_type.from_numpy(npdata)  # type: ignore

    def name(self) -> str:
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
