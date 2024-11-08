

from typing import Any
from kernelfunctions.backend import SlangModule, DeclReflection

from kernelfunctions.core.reflection import SlangProgramLayout
from kernelfunctions.function import Function
from kernelfunctions.struct import Struct
import kernelfunctions.typeregistry as tr


class ModuleFunctions:
    def __init__(self, module: 'Module'):
        super().__init__()
        self.module = module

    def __getattr__(self, name: str):
        return Function(self.module.device_module, name)


class ModuleStructs:
    def __init__(self, module: 'Module'):
        super().__init__()
        self.module = module

    def __getattr__(self, name: str):
        return Struct(self.module.device_module, name)


class Module:
    def __init__(self, device_module: SlangModule, options: dict[str, Any] = {}):
        super().__init__()
        self.device_module = device_module
        self.options = options
        self.layout = SlangProgramLayout(self.device_module.layout)

    @property
    def module(self):
        return self.device_module

    @property
    def session(self):
        return self.device_module.session

    @property
    def device(self):
        return self.session.device

    def __getattr__(self, name: str):
        with tr.scope(self.device_module):

            # Search for name as a fully qualified child struct
            slang_struct = self.layout.find_type_by_name(name)
            if slang_struct is not None:
                return Struct(self, slang_struct, options=self.options)

            # Search for name as a child of this struct
            slang_function = self.layout.find_function_by_name(name)
            if slang_function is not None:
                return Function(self.device_module, name, func_reflections=[slang_function.reflection], options=self.options)

            raise AttributeError(
                f"Type '{self.device_module.name}' has no attribute '{name}'")

    def __getitem__(self, name: str):
        return self.__getattr__(name)
