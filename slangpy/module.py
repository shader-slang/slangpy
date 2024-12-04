

from typing import TYPE_CHECKING, Any, Union
from slangpy.backend import SlangModule, ComputeKernel

from slangpy.core.reflection import SlangProgramLayout
from slangpy.function import Function
from slangpy.struct import Struct
import slangpy.typeregistry as tr

if TYPE_CHECKING:
    from slangpy.calldata import CallData
    from slangpy.dispatchdata import DispatchData


class Module:
    def __init__(self, device_module: SlangModule, options: dict[str, Any] = {}, link: list[Union['Module', SlangModule]] = []):
        super().__init__()
        assert isinstance(device_module, SlangModule)
        self.device_module = device_module
        self.options = options
        self.layout = SlangProgramLayout(self.device_module.layout)
        self.call_data_cache: dict[str, 'CallData'] = {}
        self.dispatch_data_cache: dict[str, 'DispatchData'] = {}
        self.kernel_cache: dict[str, ComputeKernel] = {}
        self.link = [x.module if isinstance(x, Module) else x for x in link]

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

    def require_struct(self, name: str):
        slang_struct = self.find_struct(name)
        if slang_struct is None:
            raise ValueError(f"Could not find struct '{name}'")
        return slang_struct

    def find_function(self, name: str):
        slang_function = self.layout.find_function_by_name(name)
        if slang_function is not None:
            res = Function()
            res.attach(module=self, func=slang_function,
                       struct=None, options=self.options)
            return res

    def require_function(self, name: str):
        slang_function = self.find_function(name)
        if slang_function is None:
            raise ValueError(f"Could not find function '{name}'")
        return slang_function

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
                res = Function()
                res.attach(module=self, func=slang_function,
                           struct=None, options=self.options)
                return res

            raise AttributeError(
                f"Type '{self.device_module.name}' has no attribute '{name}'")

    def __getitem__(self, name: str):
        return self.__getattr__(name)
