from types import NoneType
from typing import Any, Optional

from kernelfunctions.types.enums import AccessType, PrimType

from ..backend import Device

from kernelfunctions.codegen import CodeGenBlock
from kernelfunctions.typeregistry import get_or_create_type
from kernelfunctions.types.basetype import BaseType
from kernelfunctions.types.basevalue import BaseValue


class PythonFunctionCall:
    def __init__(self, *args: Any, **kwargs: Any) -> NoneType:
        super().__init__()
        self.args = [PythonValue(x, None, None) for x in args]
        self.kwargs = {n: PythonValue(v, None, n) for n, v in kwargs.items()}


class PythonValue(BaseValue):
    def __init__(self,
                 value: Any,
                 parent: Optional['PythonValue'],
                 name: Optional[str]):
        super().__init__()

        self.name = name
        self.set_type(get_or_create_type(type(value)), value)

        if isinstance(value, dict):
            self.fields = {n: PythonValue(v, self, n) for n, v in value.items()}
        else:
            self.fields = None

    def set_type(self, new_type: BaseType, value: Any = None):
        self.type = get_or_create_type(type(value))
        self.container_shape = self.type.container_shape(value)
        self.element_type = self.type.element_type(value)
        self.differentiable = self.type.differentiable(value)
        self.shape = self.type.shape(value)

    @property
    def type_name(self):
        return self.type.name

    @property
    def argument_declaration(self):
        return f"{self.type_name} {self.name}"

    def _recurse_str(self, depth: int) -> str:
        if self.fields is not None:
            child_strs = [
                f"{'  ' * depth}{name}: {child._recurse_str(depth + 1)}" for name, child in self.fields.items()]
            return "\n" + "\n".join(child_strs)
        else:
            return f"{self.name}"

    def gen_calldata(self, cgb: CodeGenBlock, name: str, access: tuple[AccessType, AccessType]):
        return self.type.gen_calldata(cgb, self, name, access)

    def gen_load(self, cgb: CodeGenBlock, from_call_data: str, to_variable: str, transform: list[Optional[int]], prim: PrimType, access: AccessType):
        return self.type.gen_load(cgb, self, from_call_data, to_variable, transform, prim, access)

    def gen_store(self, cgb: CodeGenBlock, from_variable: str, to_call_data: str, transform: list[Optional[int]], prim: PrimType, access: AccessType):
        return self.type.gen_store(cgb, self, from_variable, to_call_data, transform, prim, access)

    def create_calldata(self, device: Device, access: tuple[AccessType, AccessType], data: Any) -> Any:
        return self.type.create_calldata(device, self, access, data)

    def read_calldata(self, device: Device, access: tuple[AccessType, AccessType], data: Any, result: Any) -> None:
        return self.type.read_calldata(device, self, access, data, result)
