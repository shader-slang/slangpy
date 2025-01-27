# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING, Any, Union

from slangpy.core.function import Function
from slangpy.core.struct import Struct

from slangpy.backend import ComputeKernel, SlangModule, Device
from slangpy.core.native import NativeCallDataCache
from slangpy.reflection import SlangProgramLayout
from slangpy.bindings.typeregistry import PYTHON_SIGNATURES

import weakref

if TYPE_CHECKING:
    from slangpy.core.calldata import CallData
    from slangpy.core.dispatchdata import DispatchData

LOADED_MODULES = weakref.WeakValueDictionary()


def _check_for_hot_reload(event_info: Any = None):
    global LOADED_MODULES
    for module in LOADED_MODULES.values():
        if module is not None:
            module.on_hot_reload()


def _register_hot_reload_hook(device: Device):
    for x in LOADED_MODULES.values():
        if isinstance(x, Module):
            if (x.device == device):
                return
    device.register_shader_hot_reload_callback(_check_for_hot_reload)


class CallDataCache(NativeCallDataCache):
    def lookup_value_signature(self, o: object):
        sig = PYTHON_SIGNATURES.get(type(o))
        if sig is not None:
            return sig(o)
        else:
            return None


class Module:
    """
    A Slang module, created either by loading a slang file or providing a loaded SGL module.
    """

    def __init__(self, device_module: SlangModule, options: dict[str, Any] = {}, link: list[Union['Module', SlangModule]] = []):
        super().__init__()
        _register_hot_reload_hook(device_module.session.device)
        assert isinstance(device_module, SlangModule)
        self.device_module = device_module
        self.options = options

        #: The slangpy device module.
        self.slangpy_device_module = device_module.session.load_module('slangpy')

        #: Reflection / layout information for the module.
        self.layout = SlangProgramLayout(self.device_module.layout,
                                         self.slangpy_device_module.layout)

        self.call_data_cache = CallDataCache()
        self.dispatch_data_cache: dict[str, 'DispatchData'] = {}
        self.kernel_cache: dict[str, ComputeKernel] = {}
        self.link = [x.module if isinstance(x, Module) else x for x in link]

        self._attr_cache: dict[str, Union[Function, Struct]] = {}

        LOADED_MODULES[self.device_module.name] = self

    @staticmethod
    def load_from_source(device: Device, name: str, source: str, options: dict[str, Any] = {}, link: list[Union['Module', SlangModule]] = []):
        """
        Load a module from a string.
        """
        module = device.load_module_from_source(name, source)
        return Module(module, options=options, link=link)

    @staticmethod
    def load_from_file(device: Device, path: str, options: dict[str, Any] = {}, link: list[Union['Module', SlangModule]] = []):
        """
        Load a module from a file.
        """
        module = device.load_module(path)
        return Module(module, options=options, link=link)

    @staticmethod
    def load_from_module(device: Device, module: SlangModule, options: dict[str, Any] = {}, link: list[Union['Module', SlangModule]] = []):
        """
        Load a module from a Slang module.
        """
        return Module(module, options=options, link=link)

    @property
    def name(self):
        """
        The name of the module.
        """
        return self.device_module.name

    @property
    def module(self):
        """
        The SGL Slang module this wraps.
        """
        return self.device_module

    @property
    def session(self):
        """
        The SGL Slang session this module is part of.
        """
        return self.device_module.session

    @property
    def device(self):
        """
        The SGL device this module is part of.
        """
        return self.session.device

    def torch(self):
        """
        Returns a pytorch wrapper around this module
        """
        import slangpy.torchintegration as spytorch
        if spytorch.TORCH_ENABLED:
            return spytorch.TorchModule(self)
        else:
            raise RuntimeError("Pytorch integration is not enabled")

    def find_struct(self, name: str):
        """
        Find a struct by name, return None if not found.
        """
        slang_struct = self.layout.find_type_by_name(name)
        if slang_struct is not None:
            return Struct(self, slang_struct, options=self.options)
        else:
            return None

    def require_struct(self, name: str):
        """
        Find a struct by name, raise an error if not found.
        """
        slang_struct = self.find_struct(name)
        if slang_struct is None:
            raise ValueError(f"Could not find struct '{name}'")
        return slang_struct

    def find_function(self, name: str):
        """
        Find a function by name, return None if not found.
        """
        slang_function = self.layout.find_function_by_name(name)
        if slang_function is not None:
            res = Function(module=self, func=slang_function,
                           struct=None, options=self.options)
            return res

    def require_function(self, name: str):
        """
        Find a function by name, raise an error if not found.
        """
        slang_function = self.find_function(name)
        if slang_function is None:
            raise ValueError(f"Could not find function '{name}'")
        return slang_function

    def find_function_in_struct(self, struct: Union[Struct, str], name: str):
        """
        Find a function in a struct by name, return None if not found.
        """
        if isinstance(struct, str):
            s = self.find_struct(struct)
            if s is None:
                return None
            struct = s
        child = s.try_get_child(name)
        if child is None:
            return None
        return child.as_func()

    def on_hot_reload(self):
        """
        Called by device when the module is hot reloaded.
        """
        self.layout.on_hot_reload(self.device_module.layout)

    def __getattr__(self, name: str):
        """
        Attribute accessor attempts to find either a struct or function 
        with the specified attribute name.
        """

        # Check the cache first
        if name in self._attr_cache:
            return self._attr_cache[name]

        # Search for name as a fully qualified child struct
        slang_struct = self.find_struct(name)
        if slang_struct is not None:
            self._attr_cache[name] = slang_struct
            return slang_struct

        # Search for name as a child of this struct
        slang_function = self.layout.find_function_by_name(name)
        if slang_function is not None:
            res = Function(module=self, func=slang_function,
                           struct=None, options=self.options)
            self._attr_cache[name] = res
            return res

        raise AttributeError(
            f"Type '{self.device_module.name}' has no attribute '{name}'")

    def __getitem__(self, name: str):
        """
        Item accessor attempts to find either a struct or function
        with the specified item name (by calling __getattr__).
        """
        return self.__getattr__(name)
