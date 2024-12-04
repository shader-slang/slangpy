from typing import Any, Optional
import numpy.typing as npt
from slangpy.core.function import Function
from slangpy.core.struct import Struct
from slangpy.types.buffer import NDBuffer, NDDifferentiableBuffer


class InstanceList:
    def __init__(self, struct: Struct, data: Optional[Any] = None):
        super().__init__()
        if data is None:
            data = {}
        self._loaded_functions: dict[str, Function] = {}
        self.set_data(data)
        self._struct = struct
        self._init = self._try_load_func("__init")

    def set_data(self, data: Any):
        self._data = data

    def get_this(self) -> Any:
        return self._data

    def update_this(self, value: Any) -> None:
        pass

    def construct(self, *args: Any, **kwargs: Any) -> Any:
        if self._init is not None:
            self._init(*args, **kwargs, _result=self.get_this())

    def __getattr__(self, name: str) -> Any:
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
        if name in ['_data', '_struct', '_loaded_functions', '_init']:
            return super().__setattr__(name, value)
        if isinstance(self._data, dict) and self._struct.struct.fields is not None and name in self._struct.struct.fields:
            self._data[name] = value
        return super().__setattr__(name, value)

    def _try_load_func(self, name: str):
        func = self._struct.try_get_child(name)
        if isinstance(func, Function):
            if name != "__init":
                func = func.bind(self)
            self._loaded_functions[name] = func
            return func
        else:
            return None


class InstanceListBuffer(InstanceList):
    def __init__(self, struct: Struct, shape: tuple[int, ...], data: Optional[NDBuffer] = None):
        if data is None:
            data = NDBuffer(struct.device_module.session.device,
                            element_type=struct, shape=shape)
        super().__init__(struct, data)
        if data is None:
            data = {}

    @property
    def shape(self):
        return self._data.shape

    @property
    def buffer(self) -> NDBuffer:
        return self._data

    def to_numpy(self):
        return self.buffer.to_numpy()

    def from_numpy(self, data: npt.ArrayLike):
        self.buffer.from_numpy(data)


class InstanceListDifferentiableBuffer(InstanceList):
    def __init__(self, struct: Struct, shape: tuple[int, ...], data: Optional[NDDifferentiableBuffer] = None):
        if data is None:
            data = NDDifferentiableBuffer(struct.device_module.session.device,
                                          element_type=struct, shape=shape, requires_grad=True)
        super().__init__(struct, data)
        if data is None:
            data = {}

    @property
    def shape(self):
        return self._data.shape

    @property
    def buffer(self):
        return self._data

    def primal_to_numpy(self):
        return self.buffer.primal_to_numpy()

    def primal_from_numpy(self, data: npt.ArrayLike):
        self.buffer.primal_from_numpy(data)

    def grad_to_numpy(self):
        return self.buffer.grad_to_numpy()

    def grad_from_numpy(self, data: npt.ArrayLike):
        self.buffer.grad_from_numpy(data)
