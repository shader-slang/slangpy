
"""
This file contains python-only emulation for the current native functionality of slangpy embedded in SGL
"""

from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, cast

if TYPE_CHECKING:
    from . import Device


class AccessType(Enum):
    none = 0
    read = 1
    write = 2
    readwrite = 3


TLooseShape = tuple[Optional[int], ...]
TConcreteShape = tuple[int, ...]


class NativeBoundVariableException(Exception):
    """
    Native base class for all variable exceptions
    """

    def __init__(self, message: str, source: Optional['NativeBoundVariableRuntime'] = None):
        super().__init__(message)
        self.message = message
        self.source = source


class NativeShape:
    """
    Native base class for all shapes
    """

    def __init__(self, shape: Optional[Union[tuple[int, ...], 'NativeShape']] = None):
        super().__init__()
        if shape is None:
            self.shape: tuple[int, ...] = ()
        elif isinstance(shape, tuple):
            self.shape = shape
        else:
            self.shape = shape.shape

    def __add__(self, other: 'NativeShape') -> 'NativeShape':
        return NativeShape(self.shape + other.shape)


class NativeType:
    """
    Native base class for all type marshalls
    """

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        raise NotImplementedError()

    @property
    def element_type(self) -> 'NativeType':
        raise NotImplementedError()

    def get_byte_size(self, value: Any = None) -> int:
        raise NotImplementedError()

    def get_container_shape(self, value: Any = None) -> TLooseShape:
        return ()

    def get_shape(self, value: Any = None) -> TLooseShape:
        return self.get_container_shape(value) + self.element_type.get_shape()

    def create_calldata(self, context: 'CallContext', binding: 'NativeBoundVariableRuntime', data: Any) -> Any:
        pass

    def read_calldata(self, context: 'CallContext', binding: 'NativeBoundVariableRuntime', data: Any, result: Any) -> None:
        pass

    def create_output(self, context: 'CallContext') -> Any:
        raise NotImplementedError()

    def read_output(self, context: 'CallContext', data: Any) -> Any:
        raise NotImplementedError()


class NativeBoundCallRuntime:
    """
    Native base for BoundCallRuntime
    """

    def __init__(self):
        super().__init__()
        self.args: list['NativeBoundVariableRuntime'] = []
        self.kwargs: dict[str, 'NativeBoundVariableRuntime'] = {}

    def calculate_call_shape(self, call_dimensionality: int, *args: Any, **kwargs: Any):

        call_shape = [1] * call_dimensionality
        sig_args = self.args
        sig_kwargs = self.kwargs

        for idx, value in enumerate(args):
            sig_args[idx].populate_call_shape(call_shape, value)

        for key, value in kwargs.items():
            sig_kwargs[key].populate_call_shape(call_shape, value)

        return cast(TConcreteShape, tuple(call_shape))

    def write_calldata_pre_dispatch(self, context: 'CallContext', call_data: dict[str, Any], *args: Any, **kwargs: Any):
        """
        Write the call data for args + kwargs before dispatching
        """
        sig_args = self.args
        sig_kwargs = self.kwargs

        for idx, value in enumerate(args):
            sig_args[idx].write_call_data_pre_dispatch(context, call_data, value)

        for key, value in kwargs.items():
            sig_kwargs[key].write_call_data_pre_dispatch(context, call_data, value)

    def read_call_data_post_dispatch(self, context: 'CallContext', call_data: dict[str, Any], *args: Any, **kwargs: Any):
        """
        Read the call data for args + kwargs after dispatching
        """
        sig_args = self.args
        sig_kwargs = self.kwargs

        for idx, value in enumerate(args):
            sig_args[idx].read_call_data_post_dispatch(context, call_data, value)
        for key, value in kwargs.items():
            sig_kwargs[key].read_call_data_post_dispatch(context, call_data, value)


class NativeBoundVariableRuntime:
    """
    Native base for BoundVariableRuntime
    """

    def __init__(self):
        super().__init__()
        self.access: tuple[AccessType, AccessType] = (AccessType.none, AccessType.none)
        self.transform: Optional[TConcreteShape] = None
        self.slang_shape: TLooseShape = ()
        self.python_type: NativeType = None  # type: ignore
        self.shape: TConcreteShape = ()
        self._name = ""
        self._variable_name = ""
        self._children: Optional[dict[str, 'NativeBoundVariableRuntime']] = None

    def populate_call_shape(self, call_shape: list[int], value: Any):
        """
        Recursively calculate call shape for the node
        """
        if self._children is not None:
            for name, child in self._children.items():
                child.populate_call_shape(call_shape, value[name])
        elif value is not None:
            # Get concrete primal shape
            shape = cast(TConcreteShape, self.python_type.get_shape(value))
            tf = cast(TConcreteShape, self.transform)
            csl = len(call_shape)
            self.shape = shape

            for i in range(len(tf)):
                # Get value shape and corresponding index in the overall call shape
                shape_dim = cast(int, shape[i])
                call_idx = cast(int, tf[i])

                # Not interested in dimensionality for sub-kernel elements
                if call_idx >= csl:
                    continue

                # Apply shape, failing if we find mismatch
                cs = call_shape[call_idx]
                if cs != shape_dim:
                    if cs != 1 and shape_dim != 1:
                        raise NativeBoundVariableException(
                            f"Shape mismatch for {self._variable_name} between input and output", self)
                    if shape_dim != 1:
                        call_shape[call_idx] = shape_dim

    def write_call_data_pre_dispatch(self, context: 'CallContext', call_data: dict[str, Any], value: Any):
        """Writes value to call data dictionary pre-dispatch"""
        if self._children is not None:
            res = {}
            for name, child in self._children.items():
                child.write_call_data_pre_dispatch(context, res, value[name])
            if len(res) > 0:
                call_data[self._variable_name] = res
        else:
            # Get concrete primal shape
            shape = self.shape

            # Get call shape + append slang primal shape
            full_cs = context.call_shape + self.slang_shape

            # Broadcast occurs if the shape of the input is different from the shape of the output
            broadcast = []
            transform = cast(TConcreteShape, self.transform)
            for i in range(len(transform)):
                csidx = transform[i]
                broadcast.append(full_cs[csidx] != shape[i])

            cd_val = self.python_type.create_calldata(
                context, self, value)
            if cd_val is not None:
                call_data[self._variable_name] = cd_val

    def read_call_data_post_dispatch(self, context: 'CallContext', call_data: dict[str, Any], value: Any):
        """Reads value from call data dictionary post-dispatch"""
        if self._children is not None:
            cd_val = call_data.get(self._variable_name, None)
            for name, child in self._children.items():
                if child._variable_name in cd_val:
                    child.read_call_data_post_dispatch(context, cd_val, value[name])
        else:
            cd_val = call_data.get(self._variable_name, None)
            if cd_val is not None:
                self.python_type.read_calldata(context, self, value, cd_val)

    def read_output(self, context: 'CallContext', data: Any):
        """Reads output from function for a return value"""
        if self._children is not None:
            assert isinstance(data, dict)
            res = {}
            for name, child in self._children.items():
                child_data = data.get(child._name, None)
                if child_data is not None:
                    res[name] = child.read_output(context, child_data)
            return res
        else:
            if self.access[0] in [AccessType.write, AccessType.readwrite]:
                return self.python_type.read_output(context, data)


class NativeCallData:
    """
    Native base for CallData
    """

    def __init__(self):
        super().__init__()


class CallContext:
    """
    Native call context
    """

    def __init__(self, device: 'Device', call_shape: TConcreteShape):
        super().__init__()
        self.device = device
        self.call_shape = call_shape


def hash_signature(value_to_id: Callable[[Any], str], *args: Any, **kwargs: Any) -> str:
    """
    Generates a unique hash for a given python signature
    """

    x = []

    x.append("args\n")
    for arg in args:
        x.append(f"N:")
        _get_value_signature(value_to_id, arg, x)

    x.append("kwargs\n")
    for k, v in kwargs.items():
        x.append(f"{k}:")
        _get_value_signature(value_to_id, v, x)

    text = "".join(x)
    return text


def _get_value_signature(value_to_id: Callable[[Any], str], x: Any, out: list[str]):
    """
    Recursively get the signature of x
    """

    out.append(type(x).__name__)

    s = getattr(x, "get_this", None)
    if s is not None:
        _get_value_signature(value_to_id, s(), out)
        return

    s = getattr(x, "slangpy_signature", None)
    if s is not None:
        out.append(s)
        out.append("\n")
        return

    if isinstance(x, dict):
        out.append("\n")
        for k, v in x.items():
            out.append(f"{k}:")
            _get_value_signature(value_to_id, v, out)
        return

    s = value_to_id(x)
    if s is not None:
        out.append(s)
    out.append("\n")
