# SPDX-License-Identifier: Apache-2.0
from typing import Any, Optional, Union

from slangpy.core.function import Function
from slangpy.core.struct import Struct
from slangpy.torchintegration.torchfunction import TorchFunction, check_cuda_enabled


class TorchStruct:
    """
    A Slang struct, typically created by accessing it via a module or parent struct. i.e. mymodule.Foo,
    or mymodule.Foo.Bar. 
    """

    def __init__(self, struct: Struct) -> None:
        super().__init__()
        check_cuda_enabled(struct.module.device)
        self.struct = struct

    @property
    def name(self) -> str:
        """
        The name of the struct.
        """
        return self.struct.name

    @property
    def session(self):
        """
        The Slang session the struct's module belongs to.
        """
        return self.struct.session

    @property
    def device(self):
        """
        The device the struct's module belongs to.
        """
        return self.struct.device

    @property
    def device_module(self):
        """
        The Slang module the struct belongs to.
        """
        return self.struct.device_module

    def try_get_child(self, name: str) -> Optional[Union['TorchStruct', 'TorchFunction']]:
        """
        Attempt to get either a child struct or method of this struct.
        """
        spy_res = self.struct.try_get_child(name)
        if isinstance(spy_res, Struct):
            return TorchStruct(spy_res)
        if isinstance(spy_res, Function):
            return TorchFunction(spy_res)
        return None

    def __getattr__(self, name: str) -> Union['TorchStruct', 'TorchFunction']:
        """
        Get a child struct or method of this struct.
        """
        child = self.try_get_child(name)
        if child is not None:
            return child
        raise AttributeError(f"Type '{self.name}' has no attribute '{name}'")

    def __getitem__(self, name: str):
        """
        Get a child struct or method of this struct.
        """
        return self.__getattr__(name)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise AttributeError(f"Type '{self.name}' is not callable")

    def as_func(self) -> 'TorchFunction':
        """
        Typing helper to detect attempting to treat the struct as a function.
        """
        raise ValueError("Cannot convert a struct to a function")

    def as_struct(self) -> 'TorchStruct':
        """
        Typing helper to cast the struct to struct (no-op).
        """
        return self
