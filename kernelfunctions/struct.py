from typing import TYPE_CHECKING, Any, Optional, Union
from kernelfunctions.function import Function
from kernelfunctions.utils import try_find_function_overloads_via_ast

if TYPE_CHECKING:
    from kernelfunctions import Module
    from kernelfunctions.core.reflection import SlangType


class Struct:
    def __init__(self, module: 'Module', slang_struct: 'SlangType', options: dict[str, Any] = {}) -> None:
        super().__init__()
        self.module = module
        self.options = options
        self.struct = slang_struct
        self.slangpy_signature = self.struct.full_name

    @property
    def name(self) -> str:
        return self.struct.full_name

    @property
    def session(self):
        return self.module.device_module.session

    @property
    def device(self):
        return self.session.device

    @property
    def device_module(self):
        return self.module.device_module

    def try_get_child(self, name: str) -> Optional[Union['Struct', 'Function']]:

        # First try to find the child using the search functions in the reflection API

        # Search for name as a fully qualified child struct
        name_if_struct = f"{self.name}::{name}"
        slang_struct = self.module.layout.find_type_by_name(name_if_struct)
        if slang_struct is not None:
            return Struct(self.module, slang_struct, options=self.options)

        # Search for name as a child of this struct
        if name == "__init":
            name = "$init"
        slang_function = self.module.layout.find_function_by_name_in_type(
            self.struct, name)
        if slang_function is not None:
            return Function(self.module, self, slang_function, options=self.options)

        # Currently have Slang issue finding the init function, so for none-generic classes,
        # try to find it via the AST.
        if not '<' in self.name and name == "$init":
            (type, funcs) = try_find_function_overloads_via_ast(
                self.device_module.module_decl, self.name, name)
            if funcs is not None and len(funcs) > 0:
                return Function(self.module, self, funcs, options=self.options)

        return None

    def __getattr__(self, name: str) -> Union['Struct', 'Function']:
        child = self.try_get_child(name)
        if child is not None:
            return child
        raise AttributeError(f"Type '{self.name}' has no attribute '{name}'")

    def __getitem__(self, name: str):
        return self.__getattr__(name)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise AttributeError(f"Type '{self.name}' is not callable")

    def as_func(self) -> 'Function':
        raise ValueError("Cannot convert a struct to a function")

    def as_struct(self) -> 'Struct':
        return self
