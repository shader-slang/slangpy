from typing import Any, Callable, Optional

from kernelfunctions.backend import TypeReflection

from kernelfunctions.backend import SlangModule
from kernelfunctions.function import Function
from kernelfunctions.typeregistry import get_or_create_type
from kernelfunctions.utils import try_find_function_overloads_via_ast, try_find_type_via_ast


class Struct:
    def __init__(self, device_module: SlangModule, name: str, type_reflection: Optional[TypeReflection] = None) -> None:
        super().__init__()
        self.device_module = device_module
        self.name = name
        if type_reflection is None:
            type_reflection = self.device_module.layout.find_type_by_name(name)
        if type_reflection is None:
            raise ValueError(f"Type '{name}' not found in module {device_module.name}")
        self.type = get_or_create_type(type_reflection)

    def __getattr__(self, name: str):

        if not '<' in self.name and not '<' in name:
            # Neither or we are a generic, so attempt to use the ast to find a type decl
            name_if_struct = f"{self.name}::{name}"
            type = try_find_type_via_ast(self.device_module.module_decl, name_if_struct)
            if type is not None:
                return Struct(self.device_module, name_if_struct, type)

            # If not found, do similar searching for a global function
            if name == "__init":
                name = "$init"
            (type, funcs) = try_find_function_overloads_via_ast(
                self.device_module.module_decl, self.name, name)
            if funcs is not None and len(funcs) > 0:
                return Function(self.device_module, name, type_reflection=type, func_reflections=funcs)
        else:
            # Search for name as a fully qualified child struct
            name_if_struct = f"{self.name}::{name}"
            slang_struct = self.device_module.layout.find_type_by_name(name_if_struct)
            if slang_struct is not None:
                return Struct(self.device_module, name_if_struct, slang_struct)

            # Search for name as a child of this struct
            if name == "__init":
                name = "$init"
            parent_slang_struct = self.device_module.layout.find_type_by_name(self.name)
            slang_function = self.device_module.layout.find_function_by_name_in_type(
                parent_slang_struct, name)
            if slang_function is not None:
                return Function(self.device_module, name, type_reflection=parent_slang_struct, func_reflections=[slang_function])

        raise AttributeError(f"Type '{self.name}' has no attribute '{name}'")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise AttributeError(f"Type '{self.name}' is not callable")
