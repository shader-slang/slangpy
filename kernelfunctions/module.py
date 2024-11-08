

from typing import Any, Union
from kernelfunctions.backend import SlangModule

from kernelfunctions.core.reflection import SlangProgramLayout
from kernelfunctions.function import Function
from kernelfunctions.struct import Struct
import kernelfunctions.typeregistry as tr


class Module:
    def __init__(self, device_module: SlangModule, options: dict[str, Any] = {}):
        super().__init__()
        assert isinstance(device_module, SlangModule)
        self.device_module = device_module
        self.options = options
        self.layout = SlangProgramLayout(self.device_module.layout)

    @property
    def name(self):
        return self.device_module.name

    @property
    def module(self):
        return self.device_module

    @property
    def session(self):
        return self.device_module.session

    @property
    def device(self):
        return self.session.device

    def find_struct(self, name: str):
        slang_struct = self.layout.find_type_by_name(name)
        if slang_struct is not None:
            return Struct(self, slang_struct, options=self.options)
        else:
            return None

    def find_function(self, name: str):
        slang_function = self.layout.find_function_by_name(name)
        if slang_function is not None:
            return Function(self, None, slang_function, options=self.options)

    def find_function_in_struct(self, struct: Union[Struct, str], name: str):
        if isinstance(struct, str):
            s = self.find_struct(struct)
            if s is None:
                return None
            struct = s
        child = s.try_get_child(name)
        if child is None:
            return None
        return child.as_func()

    def __getattr__(self, name: str):
        with tr.scope(self.device_module):

            # Search for name as a fully qualified child struct
            slang_struct = self.find_struct(name)
            if slang_struct is not None:
                return slang_struct

            # Search for name as a child of this struct
            slang_function = self.layout.find_function_by_name(name)
            if slang_function is not None:
                return Function(self, None, slang_function, options=self.options)

            raise AttributeError(
                f"Type '{self.device_module.name}' has no attribute '{name}'")

    def __getitem__(self, name: str):
        return self.__getattr__(name)
