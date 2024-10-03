
"""
This file contains python-only emulation for the current native functionality of slangpy embedded in SGL.
To serve accurately, it should only import typing and the necessary backend types.
"""

from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, cast

from . import uint3, CommandBuffer

if TYPE_CHECKING:
    from . import Device, ComputeKernel


class AccessType(Enum):
    none = 0
    read = 1
    write = 2
    readwrite = 3


class CallMode(Enum):
    prim = 0
    bwds = 1
    fwds = 2


TDispatchHook = Callable[[dict[str, Any]], None]


class NativeBoundVariableException(Exception):
    """
    Native base class for all variable exceptions
    """

    def __init__(self, message: str, source: Optional['NativeBoundVariableRuntime'] = None):
        super().__init__(message)
        self.message = message
        self.source = source


class Shape:
    """
    Native shape
    """

    def __init__(self, *args: Union[None, int, 'Shape', tuple[int, ...]]):
        super().__init__()
        if len(args) == 0:
            self.shape = ()
        elif len(args) == 1:
            if args[0] == None:
                self.shape = None
            elif isinstance(args[0], tuple):
                self.shape = args[0]
            elif isinstance(args[0], Shape):
                self.shape = args[0].shape
            else:
                self.shape = self._from_tuple(args)
        else:
            self.shape = self._from_tuple(args)

    def __add__(self, other: 'Shape') -> 'Shape':
        return Shape(self.as_tuple() + other.as_tuple())

    def _from_tuple(self, shape: Any) -> tuple[int, ...]:
        if not isinstance(shape, tuple):
            raise ValueError("Shape must be a tuple")
        if any([not isinstance(x, int) for x in shape]):
            raise ValueError("Shape must be a tuple of integers")
        return shape

    def as_tuple(self) -> tuple[int, ...]:
        if self.shape is None:
            raise ValueError("Shape is invalid")
        return self.shape

    def as_list(self) -> list[int]:
        return list(self.as_tuple())

    @property
    def valid(self) -> bool:
        return self.shape is not None

    def __len__(self) -> int:
        return len(self.as_tuple())

    def __getitem__(self, key: int) -> int:
        return self.as_tuple()[key]

    def __eq__(self, value: object) -> bool:
        if isinstance(value, Shape):
            return self.shape == value.shape
        else:
            return self.shape == value


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

    def get_container_shape(self, value: Any = None) -> Shape:
        return Shape()

    def get_shape(self, value: Any = None) -> Shape:
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

        return Shape(tuple(call_shape))

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
        self.transform: Shape = None  # type: ignore
        self.slang_shape: Shape = None  # type: ignore
        self.python_type: NativeType = None  # type: ignore
        self.shape: Shape = None  # type: ignore
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
            shape = self.python_type.get_shape(value)
            tf = cast(Shape, self.transform)
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
            transform = self.transform
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
        self.device: Device = None  # type: ignore
        self.kernel: ComputeKernel = None  # type: ignore
        self.call_dimensionality = 0
        self.runtime: NativeBoundCallRuntime = None  # type: ignore
        self.sets: dict[str, Any] = {}
        self.call_mode = CallMode.prim
        self.before_dispatch_hooks: Optional[list[TDispatchHook]] = None
        self.after_dispatch_hooks: Optional[list[TDispatchHook]] = None
        self.last_call_shape = Shape()

    def call(self, *args: Any, **kwargs: Any):
        return self.exec(None, *args, **kwargs)

    def append_to(self, command_buffer: CommandBuffer, *args: Any, **kwargs: Any):
        return self.exec(command_buffer, *args, **kwargs)

    def exec(self, command_buffer: Optional[CommandBuffer],  *args: Any, **kwargs: Any):

        call_data = {}
        device = self.device
        rv_node = None

        # Build 'unpacked' args (that handle IThis)
        unpacked_args = tuple([unpack_arg(x) for x in args])
        unpacked_kwargs = {k: unpack_arg(v) for k, v in kwargs.items()}

        # Calculate call shape
        call_shape = self.runtime.calculate_call_shape(
            self.call_dimensionality, *unpacked_args, **unpacked_kwargs)
        self.last_call_shape = call_shape

        # Setup context
        context = CallContext(device, call_shape)

        # Allocate a return value if not provided in kw args
        # This is redundant if command buffer supplied, as we don't return anything
        if command_buffer is None:
            rv_node = self.runtime.kwargs.get("_result", None)
            if self.call_mode == CallMode.prim and rv_node is not None and kwargs.get("_result", None) is None:
                kwargs["_result"] = rv_node.python_type.create_output(context)
                unpacked_kwargs["_result"] = kwargs["_result"]
                rv_node.populate_call_shape(call_shape.as_list(), kwargs["_result"])

        self.runtime.write_calldata_pre_dispatch(context,
                                                 call_data, *unpacked_args, **unpacked_kwargs)

        total_threads = 1
        strides = []
        for dim in reversed(call_shape):
            strides.append(total_threads)
            total_threads *= dim
        strides.reverse()

        if len(strides) > 0:
            call_data["_call_stride"] = strides
            call_data["_call_dim"] = call_shape.as_list()
        call_data["_thread_count"] = uint3(total_threads, 1, 1)

        vars = self.sets.copy()
        vars['call_data'] = call_data

        if self.before_dispatch_hooks is not None:
            for hook in self.before_dispatch_hooks:
                hook(vars)

        # Dispatch the kernel.
        self.kernel.dispatch(uint3(total_threads, 1, 1), vars, command_buffer)

        # If just adding to command buffer, post dispatch is redundant
        if command_buffer is not None:
            return

        if self.after_dispatch_hooks is not None:
            for hook in self.after_dispatch_hooks:
                hook(vars)

        self.runtime.read_call_data_post_dispatch(
            context, call_data, *unpacked_args, **unpacked_kwargs)

        # Push updated 'this' values back to original objects
        for (i, arg) in enumerate(args):
            pack_arg(arg, unpacked_args[i])
        for (k, arg) in kwargs.items():
            pack_arg(arg, unpacked_kwargs[k])

        if self.call_mode == CallMode.prim and rv_node is not None:
            return rv_node.read_output(context, kwargs["_result"])
        else:
            return None


class CallContext:
    """
    Native call context
    """

    def __init__(self, device: 'Device', call_shape: Shape):
        super().__init__()
        self.device = device
        self.call_shape = call_shape


def unpack_arg(arg: Any) -> Any:
    if hasattr(arg, "get_this"):
        arg = arg.get_this()
    if isinstance(arg, dict):
        arg = {k: unpack_arg(v) for k, v in arg.items()}
    if isinstance(arg, list):
        arg = [unpack_arg(v) for v in arg]
    return arg


def pack_arg(arg: Any, unpacked_arg: Any):
    if hasattr(arg, "update_this"):
        arg.update_this(unpacked_arg)
    if isinstance(arg, dict):
        for k, v in arg.items():
            pack_arg(v, unpacked_arg[k])
    if isinstance(arg, list):
        for i, v in enumerate(arg):
            pack_arg(v, unpacked_arg[i])
    return arg


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
