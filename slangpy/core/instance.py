# SPDX-License-Identifier: Apache-2.0
from typing import Any, Optional

import numpy.typing as npt

from slangpy.core.function import FunctionNode
from slangpy.core.struct import Struct

from slangpy.types.buffer import NDBuffer


class InstanceList:
    """
    Represents a list of instances of a struct, either as a single buffer
    or an SOA style set of buffers for each field. data can either
    be a dictionary of field names to buffers, or a single buffer.
    """

    def __init__(self, struct: Struct, data: Optional[Any] = None):
        super().__init__()
        if data is None:
            data = {}
        self._loaded_functions: dict[str, FunctionNode] = {}
        self.set_data(data)
        self._struct = struct
        self._init = self._try_load_func("__init")

    def set_data(self, data: Any):
        """
        Set the data for the instance list. data can either
        be a dictionary of field names to buffers, or a single buffer.
        """
        self._data = data

    def get_this(self) -> Any:
        """
        IThis protocol implementation of get_this.
        """
        return self._data

    def update_this(self, value: Any) -> None:
        """
        IThis protocol implementation of update_this.
        """
        pass

    def construct(self, *args: Any, **kwargs: Any) -> Any:
        """
        Call the constructor of the struct on all elements in list.
        """
        if self._init is not None:
            self._init(*args, **kwargs, _result=self.get_this())

    def __getattr__(self, name: str) -> Any:
        """
        Either returns field data (if this is an SOA list and the field has
        been specified), or a function bound to this instance list that can becalled.
        """
        if name in self._loaded_functions:
            return self._loaded_functions[name]

        if isinstance(self._data, dict) and self._struct.struct.fields is not None and name in self._struct.struct.fields:
            return self._data.get(name)

        func = self._try_load_func(name)
        if func is not None:
            return func

        raise AttributeError(
            f"Instance of '{self._struct.name}' has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set field data (if this is an SOA list).
        """
        if name in ['_data', '_struct', '_loaded_functions', '_init']:
            return super().__setattr__(name, value)
        if isinstance(self._data, dict) and self._struct.struct.fields is not None and name in self._struct.struct.fields:
            self._data[name] = value
        return super().__setattr__(name, value)

    def _try_load_func(self, name: str):
        func = self._struct.try_get_child(name)
        if isinstance(func, FunctionNode):
            if name != "__init":
                func = func.bind(self)
            self._loaded_functions[name] = func
            return func
        else:
            return None


class InstanceBuffer(InstanceList):
    """
    Simplified implementation of InstanceList that uses a single buffer for all instances and
    provides buffer convenience functions for accessing its data.
    """

    def __init__(self, struct: Struct, shape: tuple[int, ...], data: Optional[NDBuffer] = None):
        if data is None:
            data = NDBuffer(struct.device_module.session.device,
                            dtype=struct, shape=shape)
        super().__init__(struct, data)
        if data is None:
            data = {}

    @property
    def shape(self):
        """
        Get the shape of the buffer.
        """
        return self._data.shape

    @property
    def buffer(self) -> NDBuffer:
        """
        Get the buffer.
        """
        return self._data

    def to_numpy(self):
        """
        Convert the buffer to a numpy array.
        """
        return self.buffer.to_numpy()

    def from_numpy(self, data: npt.ArrayLike):
        """
        Set the buffer from a numpy array.
        """
        self.buffer.from_numpy(data)
