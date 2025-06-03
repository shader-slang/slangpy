# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import TYPE_CHECKING, Any, Optional, Union

from slangpy.core.function import Function

if TYPE_CHECKING:
    from slangpy import Module
    from slangpy.reflection import SlangType


class Struct:
    """
    A Slang struct, typically created by accessing it via a module or parent struct. i.e. mymodule.Foo,
    or mymodule.Foo.Bar.
    """

    def __init__(
        self, module: "Module", slang_struct: "SlangType", options: dict[str, Any] = {}
    ) -> None:
        super().__init__()
        self.module = module
        self.options = options
        self.struct = slang_struct
        self.slangpy_signature = self.struct.full_name

    @property
    def program(self):
        """
        Program layout struct is part of.
        """
        return self.struct.program

    @property
    def name(self) -> str:
        """
        The name of the struct.
        """
        return self.struct.name

    @property
    def full_name(self) -> str:
        """
        The name of the struct.
        """
        return self.struct.full_name

    @property
    def element_type(self):
        """
        The element type of the struct.
        """
        e = self.struct.element_type
        if e is None:
            return None
        return Struct(self.module, e, options=self.options)

    @property
    def fields(self):
        """
        The fields of the struct.
        """
        return self.struct.fields

    @property
    def differentiable(self):
        """
        The differentiable of the struct.
        """
        return self.struct.differentiable

    @property
    def derivative(self):
        """
        The derivative of the struct.
        """
        d = self.struct.derivative
        if d is None:
            return None
        return Struct(self.module, d, options=self.options)

    @property
    def num_dims(self):
        """
        The number of dimensions of the struct.
        """
        return self.struct.num_dims

    @property
    def shape(self):
        """
        The shape of the struct.
        """
        return self.struct.shape

    @property
    def uniform_layout(self):
        """
        The uniform layout of the struct.
        """
        return self.struct.uniform_layout

    @property
    def buffer_layout(self):
        """
        The buffer layout of the struct.
        """
        return self.struct.buffer_layout

    @property
    def session(self):
        """
        The Slang session the struct's module belongs to.
        """
        return self.module.device_module.session

    @property
    def type_reflection(self):
        """
        The type reflection of the struct.
        """
        return self.struct.type_reflection

    @property
    def device(self):
        """
        The device the struct's module belongs to.
        """
        return self.session.device

    @property
    def device_module(self):
        """
        The Slang module the struct belongs to.
        """
        return self.module.device_module

    def torch(self):
        """
        Returns a pytorch wrapper around this struct
        """
        from slangpy.torchintegration.torchstruct import TorchStruct

        return TorchStruct(self)

    def try_get_child(self, name: str) -> Optional[Union["Struct", "Function"]]:
        """
        Attempt to get either a child struct or method of this struct.
        """

        # First try to find the child using the search functions in the reflection API

        # Search for name as a fully qualified child struct
        name_if_struct = f"{self.name}::{name}"
        slang_struct = self.module.layout.find_type_by_name(name_if_struct)
        if slang_struct is not None:
            return Struct(self.module, slang_struct, options=self.options)

        # Search for name as a child of this struct
        if name == "__init":
            name = "$init"
        slang_function = self.module.layout.find_function_by_name_in_type(self.struct, name)
        if slang_function is not None:
            res = Function(
                module=self.module,
                func=slang_function,
                struct=self,
                options=self.options,
            )
            return res

        return None

    def __getattr__(self, name: str) -> Union["Struct", "Function"]:
        """
        Get a child struct or method of this struct.
        """
        child = self.try_get_child(name)
        if child is not None:
            return child

        raise AttributeError(f"Struct '{self.name}' has no method or sub-type named '{name}'")

    def __getitem__(self, name: str):
        """
        Get a child struct or method of this struct.
        """
        return self.__getattr__(name)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.__getattr__("__init")(*args, **kwds)

    def as_func(self) -> "Function":
        """
        Typing helper to detect attempting to treat the struct as a function.
        """
        raise ValueError("Cannot convert a struct to a function")

    def as_struct(self) -> "Struct":
        """
        Typing helper to cast the struct to struct (no-op).
        """
        return self
