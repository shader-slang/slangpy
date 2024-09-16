import re
from types import NoneType
from typing import Any, Optional, Sequence

from kernelfunctions.types.basevalue import BaseVariable
from kernelfunctions.types.basevalueimpl import BaseVariableImpl
from kernelfunctions.types.enums import AccessType

from ..backend import Device

from kernelfunctions.codegen import CodeGenBlock
from kernelfunctions.typeregistry import get_or_create_type
from kernelfunctions.types.basetype import BaseType


class PythonFunctionCall:
    def __init__(self, *args: Any, **kwargs: Any) -> NoneType:
        super().__init__()
        self.args = [PythonVariable(x, None, None) for x in args]
        self.kwargs = {n: PythonVariable(v, None, n) for n, v in kwargs.items()}


class PythonVariable(BaseVariableImpl):
    def __init__(self,
                 value: Any,
                 parent: Optional['PythonVariable'],
                 name: Optional[str]):
        super().__init__()

        self.name = name if name is not None else ""
        self.set_type(get_or_create_type(type(value), value), value)

        if isinstance(value, dict):
            self.fields = {n: PythonVariable(v, self, n) for n, v in value.items()}
        else:
            self.fields = None

    def is_compatible(self, other: 'BaseVariable') -> bool:
        if self.fields is not None:
            if other.fields is None:
                return False
            for field in self.fields:
                if field not in other.fields:
                    return False
                if not self.fields[field].is_compatible(other.fields[field]):
                    return False
            return True

        el_name = self.root_element_name
        other_name = other.root_element_name

        if el_name == other_name:
            return True
        if el_name == 'none' or other_name == 'none':
            return True

        stripped_primal_name = re.sub(
            r"\d+_t", "", el_name).replace("uint", "int")
        stripped_other_name = re.sub(
            r"\d+_t", "", other_name).replace("uint", "int")

        if stripped_primal_name == stripped_other_name:
            return True

        if stripped_primal_name == f"vector<{stripped_other_name},1>":
            return True
        if f"vector<{stripped_primal_name},1>" == stripped_other_name:
            return True

        return False

    def set_type(self, new_type: BaseType, value: Any = None):
        self.primal = new_type
        self.derivative = self.primal.differentiate(value)
        self.container_shape = self.primal.container_shape(value)
        self.element_type = self.primal.element_type(value)
        self.differentiable = self.primal.differentiable(value)
        self.has_derivative = self.primal.has_derivative(value)
        self.shape = self.primal.shape(value)
        self.writable = self.primal.is_writable(value)
        self._primal_type_name = self.primal.name(value)
        self._derivative_type_name = self.derivative.name(
            value) if self.derivative is not None else None
        self._root_element_name = self._find_bottom_level_element(value).name(value)
        self._primal_element_name = self.primal.element_type(value).name(value)
        self._derivative_element_name = self.derivative.element_type(
            value).name(value) if self.derivative is not None else None

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

    @property
    def primal_type_name(self):
        return self._primal_type_name

    @property
    def derivative_type_name(self):
        return self._derivative_type_name

    @property
    def primal_element_name(self):
        return self._primal_element_name

    @property
    def derivative_element_name(self):
        return self._derivative_element_name

    @property
    def root_element_name(self):
        return self._root_element_name
