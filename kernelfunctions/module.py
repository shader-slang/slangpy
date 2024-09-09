

from kernelfunctions.backend import SlangModule

from kernelfunctions.function import Function


class ModuleFunctions:
    def __init__(self, module: 'Module'):
        super().__init__()
        self.module = module

    def __getattr__(self, name: str):
        return Function(self.module.device_module, name)


class Module:
    def __init__(self, device_module: SlangModule):
        super().__init__()
        self.device_module = device_module
