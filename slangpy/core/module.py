# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import TYPE_CHECKING, Any, Optional, Union, Sequence

from slangpy.core.function import Function
from slangpy.core.struct import Struct

from slangpy import Pipeline, ShaderTable, SlangModule, Device, Logger
from slangpy.core.native import NativeCallDataCache
from slangpy.reflection import SlangProgramLayout
from slangpy.bindings.typeregistry import lookup_signature_callback

import weakref

if TYPE_CHECKING:
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
            if x.device == device:
                return
    device.register_shader_hot_reload_callback(_check_for_hot_reload)


class CallDataCache(NativeCallDataCache):
    def lookup_value_signature(self, o: object):
        found, sig = lookup_signature_callback(type(o))
        if not found:
            return None
        if sig is not None:
            return sig(o)
        else:
            return None


class Module:
    """
    A Slang module, created either by loading a slang file or providing a loaded SGL module.
    """

    def __init__(
        self,
        device_module: SlangModule,
        options: dict[str, Any] = {},
        link: Sequence[Union["Module", SlangModule]] = [],
    ):
        super().__init__()
        _register_hot_reload_hook(device_module.session.device)
        assert isinstance(device_module, SlangModule)
        self.options = options

        # Normalize link list to SlangModule instances
        link_slang_modules = [x.module if isinstance(x, Module) else x for x in link]

        # Load slangpy module
        self.slangpy_device_module = device_module.session.load_module("slangpy")

        # Always compose the module with slangpy and any links
        all_modules = [self.slangpy_device_module, device_module] + link_slang_modules
        composed = device_module.session.compose_modules(device_module.name, all_modules)
        self.device_module = composed
        self.layout = SlangProgramLayout(composed.layout)

        # Store link modules (excluding slangpy)
        # TODO: We should remove this, but some applications currently still rely on this.
        self.link = list(dict.fromkeys(link_slang_modules))

        self.call_data_cache = CallDataCache()
        self.dispatch_data_cache: dict[str, "DispatchData"] = {}
        self.pipeline_cache: dict[str, Pipeline] = {}
        self.shader_table_cache: dict[str, ShaderTable] = {}
        self.logger: Optional[Logger] = None

        self._attr_cache: dict[str, Union[Function, Struct]] = {}
        self._all_functions: weakref.WeakSet[Function] = weakref.WeakSet()

        LOADED_MODULES[self.device_module.name] = self

    @staticmethod
    def load_from_source(
        device: Device,
        name: str,
        source: str,
        options: dict[str, Any] = {},
        link: Sequence[Union["Module", SlangModule]] = [],
    ):
        """
        Load a module from a string.
        """
        module = device.load_module_from_source(name, source)
        return Module(module, options=options, link=link)

    @staticmethod
    def load_from_file(
        device: Device,
        path: str,
        options: dict[str, Any] = {},
        link: Sequence[Union["Module", SlangModule]] = [],
    ):
        """
        Load a module from a file.
        """
        module = device.load_module(path)
        return Module(module, options=options, link=link)

    @staticmethod
    def load_from_module(
        device: Device,
        module: SlangModule,
        options: dict[str, Any] = {},
        link: list[Union["Module", SlangModule]] = [],
    ):
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
            res = Function(module=self, func=slang_function, struct=None, options=self.options)
            self._all_functions.add(res)
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
        child = struct.try_get_child(name)
        if child is None:
            return None
        return child.as_func()

    def on_hot_reload(self):
        """
        Called by device when the module is hot reloaded.
        """
        # C++ side handles reload for composed modules; layout returns fresh combined layout.
        self.layout.on_hot_reload(self.device_module.layout)

        # Create new cache and update all tracked Function objects
        self.call_data_cache = CallDataCache()
        for func in self._all_functions:
            func._native_cache = self.call_data_cache

        # Clear all caches
        self.dispatch_data_cache = {}
        self.pipeline_cache = {}
        self.shader_table_cache = {}
        self._attr_cache = {}

    def __getattr__(self, name: str):
        """
        Attribute accessor attempts to find either a struct or function
        with the specified attribute name.
        """

        # Check the cache first
        if name in self._attr_cache:
            return self._attr_cache[name]

        # Check if it is a function first (workaround for slang #6317)
        slang_function = self.layout.find_function_by_name(name)
        if slang_function is not None:
            res = Function(module=self, func=slang_function, struct=None, options=self.options)
            self._all_functions.add(res)
            self._attr_cache[name] = res
            return res

        # Search for name as a fully qualified child struct
        slang_struct = self.find_struct(name)
        if slang_struct is not None:
            self._attr_cache[name] = slang_struct
            return slang_struct

        raise AttributeError(
            f"Module '{self.device_module.name}' has no function or type named '{name}'"
        )

    def __getitem__(self, name: str):
        """
        Item accessor attempts to find either a struct or function
        with the specified item name (by calling __getattr__).
        """
        return self.__getattr__(name)

    def __repr__(self) -> str:
        """
        Return a string representation of the Module for debugging.
        """
        return f"Module(name='{self.device_module.name}', linked_modules={len(self.link)})"
