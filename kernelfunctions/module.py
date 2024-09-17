

from kernelfunctions.backend import SlangModule, DeclReflection

from kernelfunctions.function import Function
from kernelfunctions.struct import Struct


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
    def __init__(self, device_module: SlangModule):
        super().__init__()
        self.device_module = device_module

    @property
    def module(self):
        return self.device_module

    @property
    def layout(self):
        return self.device_module.layout

    @property
    def session(self):
        return self.device_module.session

    @property
    def device(self):
        return self.session.device

    def __getattr__(self, name: str):

        if not '<' in name:
            # Not a generic, so attempt to use the ast to find a type decl
            type_decl = self.device_module.module_decl.find_first_child_of_kind(
                DeclReflection.Kind.struct, name)
            if type_decl is not None:
                return Struct(self.device_module, name, type_decl.as_type())

            # If not found, do similar searching for a global function
            func_decls = self.device_module.module_decl.find_children_of_kind(
                DeclReflection.Kind.func, name)
            if len(func_decls) > 0:
                return Function(self.device_module, name, func_reflections=[x.as_function() for x in func_decls])
        else:
            # Generics need slang's type resolution

            # Search for name as a fully qualified child struct
            slang_struct = self.device_module.layout.find_type_by_name(name)
            if slang_struct is not None:
                return Struct(self.device_module, name, slang_struct)

            # Search for name as a child of this struct
            slang_function = self.device_module.layout.find_function_by_name(name)
            if slang_function is not None:
                return Function(self.device_module, name, func_reflections=[slang_function])

        raise AttributeError(
            f"Type '{self.device_module.name}' has no attribute '{name}'")
