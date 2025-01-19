# SPDX-License-Identifier: Apache-2.0
from typing import Any, Union

from slangpy.core.function import Function
from slangpy.core.struct import Struct

from slangpy.backend import SlangModule, Device
from slangpy.core.module import Module

from slangpy.torchintegration.torchfunction import TorchFunction, check_cuda_enabled
from slangpy.torchintegration.torchstruct import TorchStruct


class TorchModule:
    """
    A Slang module, created either by loading a slang file or providing a loaded SGL module.
    """

    def __init__(self, module: 'Module'):
        super().__init__()
        check_cuda_enabled(module.device)
        self.module = module

    @staticmethod
    def load_from_source(device: Device, name: str, source: str, options: dict[str, Any] = {}, link: list[Union['Module', SlangModule]] = []):
        """
        Load a module from a string.
        """
        spy_module = Module.load_from_source(device, name, source, options, link)
        return TorchModule(spy_module)

    @staticmethod
    def load_from_file(device: Device, path: str, options: dict[str, Any] = {}, link: list[Union['Module', SlangModule]] = []):
        """
        Load a module from a file.
        """
        spy_module = Module.load_from_file(device, path, options, link)
        return TorchModule(spy_module)

    @staticmethod
    def load_from_module(device: Device, module: SlangModule, options: dict[str, Any] = {}, link: list[Union['Module', SlangModule]] = []):
        """
        Load a module from a Slang module.
        """
        spy_module = Module.load_from_module(device, module, options, link)
        return TorchModule(spy_module)

    @property
    def name(self):
        """
        The name of the module.
        """
        return self.module.name

    @property
    def session(self):
        """
        The SGL Slang session this module is part of.
        """
        return self.module.session

    @property
    def device(self):
        """
        The SGL device this module is part of.
        """
        return self.module.device

    def find_struct(self, name: str):
        """
        Find a struct by name, return None if not found.
        """
        spy_struct = self.module.find_struct(name)
        if spy_struct is not None:
            return TorchStruct(spy_struct)
        return None

    def require_struct(self, name: str):
        """
        Find a struct by name, raise an error if not found.
        """
        return TorchStruct(self.module.require_struct(name))

    def find_function(self, name: str):
        """
        Find a function by name, return None if not found.
        """
        spy_function = self.module.find_function(name)
        if spy_function is not None:
            return TorchFunction(spy_function)
        return None

    def require_function(self, name: str):
        """
        Find a function by name, raise an error if not found.
        """
        return TorchFunction(self.module.require_function(name))

    def find_function_in_struct(self, struct: Union[TorchStruct, Struct, str], name: str):
        """
        Find a function in a struct by name, return None if not found.
        """
        if isinstance(struct, TorchStruct):
            struct = struct.struct
        spy_function = self.module.find_function_in_struct(struct, name)
        if spy_function is not None:
            return TorchFunction(spy_function)
        return None

    def __getattr__(self, name: str):
        """
        Attribute accessor attempts to find either a struct or function 
        with the specified attribute name.
        """
        res = self.module.__getattr__(name)
        if isinstance(res, Struct):
            return TorchStruct(res)
        if isinstance(res, Function):
            return TorchFunction(res)
        return res

    def __getitem__(self, name: str):
        """
        Item accessor attempts to find either a struct or function
        with the specified item name (by calling __getattr__).
        """
        return self.__getattr__(name)
