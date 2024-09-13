from types import NoneType
from typing import Any, Optional, Sequence

from kernelfunctions.types.basevalue import BaseValue
from kernelfunctions.types.basevalueimpl import BaseValueImpl
from kernelfunctions.types.enums import AccessType

from ..backend import Device

from kernelfunctions.codegen import CodeGenBlock
from kernelfunctions.typeregistry import get_or_create_type
from kernelfunctions.types.basetype import BaseType


class PythonFunctionCall:
    def __init__(self, *args: Any, **kwargs: Any) -> NoneType:
        super().__init__()
        self.args = [PythonValue(x, None, None) for x in args]
        self.kwargs = {n: PythonValue(v, None, n) for n, v in kwargs.items()}


class PythonValue(BaseValueImpl):
    def __init__(self,
                 value: Any,
                 parent: Optional['PythonValue'],
                 name: Optional[str]):
        super().__init__()

        self.name = name if name is not None else ""
        self.set_type(get_or_create_type(type(value), value), value)

        if isinstance(value, dict):
            self.fields = {n: PythonValue(v, self, n) for n, v in value.items()}
        else:
            self.fields = None

    def is_compatible(self, other: 'BaseValue') -> bool:
        return str(self.primal.element_type()) == str(other.primal.element_type())

    def set_type(self, new_type: BaseType, value: Any = None):
        self.primal = new_type
        self.derivative = self.primal.differentiate(value)
        self.container_shape = self.primal.container_shape(value)
        self.element_type = self.primal.element_type(value)
        self.differentiable = self.primal.differentiable(value)
        self.shape = self.primal.shape(value)

    def gen_calldata(self, cgb: CodeGenBlock, name: str, transform: list[Optional[int]], access: tuple[AccessType, AccessType]):
        return self.primal.gen_calldata(cgb, self, name, transform, access)

    def gen_load_store(self, cgb: CodeGenBlock, name: str, transform: list[Optional[int]], access: tuple[AccessType, AccessType]):
        return self.primal.gen_load_store(cgb, self, name, transform, access)

    def create_calldata(self, device: Device, access: tuple[AccessType, AccessType], data: Any) -> Any:
        return self.primal.create_calldata(device, self, access, data)

    def read_calldata(self, device: Device, access: tuple[AccessType, AccessType], data: Any, result: Any) -> None:
        return self.primal.read_calldata(device, self, access, data, result)

    def create_output(self, device: Device, call_shape: Sequence[int]) -> Any:
        return self.primal.create_output(device, call_shape)

    def read_output(self, device: Device, data: Any) -> Any:
        return self.primal.read_output(device, data)
