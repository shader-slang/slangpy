from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, cast

from . import CommandBuffer, ModifierID, TypeReflection, uint3

"""
This file contains python-only emulation for the current native functionality of slangpy embedded in SGL.
To serve accurately, it should only import typing and the necessary backend types.
"""


if TYPE_CHECKING:
    from . import ComputeKernel, Device


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

    def __repr__(self) -> str:
        if self.valid:
            return f"Shape({self.as_tuple()})"
        else:
            return "Shape(None)"

    def __str__(self) -> str:
        if self.valid:
            return f"{self.as_tuple()}"
        else:
            return "<invalid>"

    @property
    def valid(self) -> bool:
        return self.shape is not None

    @property
    def concrete(self) -> bool:
        return self.shape is not None and -1 not in self.shape

    def __len__(self) -> int:
        return len(self.as_tuple())

    def __getitem__(self, key: int) -> int:
        if key >= len(self):
            raise IndexError("Shape index out of range")  # @IgnoreException
        return self.as_tuple()[key]

    def __eq__(self, value: object) -> bool:
        if isinstance(value, Shape):
            return self.shape == value.shape
        else:
            return self.shape == value

    def __iter__(self):
        return iter(self.as_tuple())


class NativeSlangType:
    def __init__(self):
        super().__init__()
        self._reflection: TypeReflection = None  # type: ignore


class NativeType:
    """
    Native base class for all type marshalls
    """

    def __init__(self):
        super().__init__()
        self.concrete_shape: Shape = Shape(None)

    def get_shape(self, value: Any = None) -> Shape:
        return self.concrete_shape

    def create_calldata(self, context: 'CallContext', binding: 'NativeBoundVariableRuntime', data: Any) -> Any:
        pass

    def create_dispatchdata(self, data: Any) -> Any:
        raise NotImplementedError()

    def read_calldata(self, context: 'CallContext', binding: 'NativeBoundVariableRuntime', data: Any, result: Any) -> None:
        pass

    def create_output(self, context: 'CallContext', binding: 'NativeBoundVariableRuntime') -> Any:
        raise NotImplementedError()

    def read_output(self, context: 'CallContext', binding: 'NativeBoundVariableRuntime', data: Any) -> Any:
        return None


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

    def write_raw_dispatch_data(self, call_data: dict[str, Any], **kwargs: Any):
        """
        Simplified write call data for raw dispatch
        """
        sig_kwargs = self.kwargs
        for key, value in kwargs.items():
            sig_kwargs[key].write_raw_dispatch_data(call_data, value)


class NativeBoundVariableRuntime:
    """
    Native base for BoundVariableRuntime
    """

    def __init__(self):
        super().__init__()
        self.access: tuple[AccessType, AccessType] = (AccessType.none, AccessType.none)
        self.transform: Shape = None  # type: ignore
        self.python_type: NativeType = None  # type: ignore
        self.vector_type: NativeSlangType = None  # type: ignore
        self.shape: Shape = None  # type: ignore
        self.variable_name = ""
        self.call_dimensionality = 0
        self.children: Optional[dict[str, 'NativeBoundVariableRuntime']] = None

    def populate_call_shape(self, call_shape: list[int], value: Any):
        """
        Recursively calculate call shape for the node
        """
        if self.children is not None:
            for name, child in self.children.items():
                child.populate_call_shape(call_shape, value[name])
        elif value is not None:
            # Get concrete primal shape. As it is invalid to broadcast from
            # elements, only need the container shape, not full shape.
            if self.python_type.concrete_shape.valid:
                shape = self.python_type.concrete_shape
            else:
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
                        raise ValueError({
                            'message': f"Shape mismatch for {self.variable_name} between input and output",
                            'source': self
                        })
                    if shape_dim != 1:
                        call_shape[call_idx] = shape_dim

    def write_call_data_pre_dispatch(self, context: 'CallContext', call_data: dict[str, Any], value: Any):
        """Writes value to call data dictionary pre-dispatch"""
        if self.children is not None:
            res = {}
            for name, child in self.children.items():
                child.write_call_data_pre_dispatch(context, res, value[name])
            if len(res) > 0:
                call_data[self.variable_name] = res
        else:
            # Get concrete primal shape
            shape = self.shape

            # Get call shape + append slang primal shape
            full_cs = context.call_shape

            # Broadcast occurs if the shape of the input is different from the shape of the output
            broadcast = []
            transform = self.transform
            for i in range(len(transform)):
                csidx = transform[i]
                broadcast.append(full_cs[csidx] != shape[i])

            cd_val = self.python_type.create_calldata(
                context, self, value)
            if cd_val is not None:
                call_data[self.variable_name] = cd_val

    def read_call_data_post_dispatch(self, context: 'CallContext', call_data: dict[str, Any], value: Any):
        """Reads value from call data dictionary post-dispatch"""
        if self.children is not None:
            cd_val = call_data.get(self.variable_name, None)
            for name, child in self.children.items():
                if child.variable_name in cd_val:
                    child.read_call_data_post_dispatch(context, cd_val, value[name])
        else:
            cd_val = call_data.get(self.variable_name, None)
            if cd_val is not None:
                self.python_type.read_calldata(context, self, value, cd_val)

    def read_output(self, context: 'CallContext', data: Any):
        """Reads output from function for a return value"""
        if self.children is not None:
            assert isinstance(data, dict)
            res = {}
            for name, child in self.children.items():
                child_data = data.get(child.variable_name, None)
                if child_data is not None:
                    res[name] = child.read_output(context, child_data)
            return res
        else:
            if self.access[0] in [AccessType.write, AccessType.readwrite]:
                return self.python_type.read_output(context, self, data)

    def write_raw_dispatch_data(self, call_data: dict[str, Any], value: Any):
        """Writes value to call data dictionary pre-dispatch"""
        if self.children is not None:
            res = {}
            for name, child in self.children.items():
                child.write_raw_dispatch_data(res, value[name])
            if len(res) > 0:
                call_data[self.variable_name] = res
        else:
            cd_val = self.python_type.create_dispatchdata(value)
            if cd_val is not None:
                call_data[self.variable_name] = cd_val


class NativeCallRuntimeOptions:
    def __init__(self):
        super().__init__()
        self.uniforms: Optional[list[Union[Callable[[
            'NativeCallData'], Any], dict[str, Any]]]] = None
        self.before_dispatch: Optional[list[TDispatchHook]] = None
        self.after_dispatch: Optional[list[TDispatchHook]] = None


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
        self.call_mode = CallMode.prim
        self.last_call_shape = Shape()

    def call(self, opts: 'NativeCallRuntimeOptions', *args: Any, **kwargs: Any):
        return self.exec(opts, None, *args, **kwargs)

    def append_to(self, opts: NativeCallRuntimeOptions, command_buffer: CommandBuffer, *args: Any, **kwargs: Any):
        return self.exec(opts, command_buffer, *args, **kwargs)

    def exec(self, opts: 'NativeCallRuntimeOptions', command_buffer: Optional[CommandBuffer],  *args: Any, **kwargs: Any):

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
                kwargs["_result"] = rv_node.python_type.create_output(context, rv_node)
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

        vars = {}
        if opts.uniforms is not None:
            for u in opts.uniforms:
                if isinstance(u, dict):
                    vars.update(u)
                else:
                    vars.update(u(self))
        vars['call_data'] = call_data

        if opts.before_dispatch is not None:
            for hook in opts.before_dispatch:
                hook(vars)

        # Dispatch the kernel.
        self.kernel.dispatch(uint3(total_threads, 1, 1), vars, command_buffer)

        # If just adding to command buffer, post dispatch is redundant
        if command_buffer is not None:
            return

        if opts.after_dispatch is not None:
            for hook in opts.after_dispatch:
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
