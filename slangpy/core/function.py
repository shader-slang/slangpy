# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol, Union, cast

from slangpy.core.native import (
    CallMode,
    SignatureBuilder,
    NativeCallRuntimeOptions,
    NativeFunctionNode,
    FunctionNodeType,
)

from slangpy.reflection import SlangFunction, SlangType
from slangpy import CommandEncoder, TypeConformance, uint3, Logger
from slangpy.slangpy import Shape
from slangpy.bindings.typeregistry import PYTHON_SIGNATURES

if TYPE_CHECKING:
    from slangpy.core.calldata import CallData
    from slangpy.core.module import Module
    from slangpy.core.struct import Struct

ENABLE_CALLDATA_CACHE = True


TCallHook = Callable[["Function"], None]


def _cache_value_to_id(val: Any) -> str:
    cb = PYTHON_SIGNATURES.get(type(val))
    if cb is None:
        return ""
    else:
        return cb(val)


class IThis(Protocol):
    def get_this(self) -> Any: ...

    def update_this(self, value: Any) -> None: ...


class FunctionBuildInfo:
    def __init__(self) -> None:
        super().__init__()

        # Will always be populated by the root
        self.name: str
        self.module: "Module"
        self.function: SlangFunction
        self.this_type: Optional[SlangType]

        # Optional value that will be set depending on the chain.
        self.map_args: tuple[Any, ...] = ()
        self.map_kwargs: dict[str, Any] = {}
        self.type_conformances: list[TypeConformance] = []
        self.call_mode: CallMode = CallMode.prim
        self.options: dict[str, Any] = {}
        self.constants: dict[str, Any] = {}
        self.thread_group_size: Optional[uint3] = None
        self.return_type: Optional[Union[type, str]] = None
        self.logger: Optional[Logger] = None
        self.call_group_shape: Optional[Shape] = None


class FunctionNode(NativeFunctionNode):
    @property
    def root(self):
        """
        Get the root function node
        """
        return cast(Function, self._find_native_root())

    @property
    def name(self):
        """
        Get the name of the function.
        """
        return self.root._name

    @property
    def parent(self):
        """
        Get the parent function node
        """
        return cast(FunctionNode, self._native_parent)

    @property
    def module(self):
        """
        Get the module that the function is part of
        """
        return self.root._module

    def torch(self):
        """
        Returns a pytorch wrapper around this function
        """
        pass

    def bind(self, this: IThis) -> "FunctionNode":
        """
        Bind a `this` object to the function. Typically
        this is called automatically when calling a function on a struct.
        """
        return FunctionNodeBind(self, this)

    def map(self, *args: Any, **kwargs: Any):
        """
        Apply dimension or type mapping to all or some of the arguments.

        myfunc.map((1,)(0,))(arg1, arg2) # Map arg1 to dimension 1, arg2 to dimension 0

        myfunc.map(module.Foo, module.Bar)(arg1, arg2) # Cast arg1 to Foo, arg2 to Bar
        """
        return FunctionNodeMap(self, args, kwargs)

    def set(self, *args: Any, **kwargs: Any):
        """
        Specify additional uniform values that should be set whenever the function's kernel
        is dispatched. Useful for setting constants or other values that are not passed as arguments.
        """
        if len(args) > 0 and len(kwargs) > 0:
            raise ValueError("Set accepts either positional or keyword arguments, not both")
        if len(args) > 1:
            raise ValueError("Set accepts only one positional argument (a dictionary or callback)")
        if len(kwargs) > 0:
            return FunctionNodeSet(self, kwargs)
        elif len(args) > 0 and (callable(args[0]) or isinstance(args[0], dict)):
            return FunctionNodeSet(self, args[0])
        else:
            raise ValueError(
                "Set requires either keyword arguments or 1 dictionary / hook argument"
            )

    def constants(self, constants: dict[str, Any]):
        """
        Specify link time constants that should be set when the function is compiled. These are
        the most optimal way of specifying unchanging data, however note that changing a constant
        will result in the function being recompiled.
        """
        return FunctionNodeConstants(self, constants)

    def type_conformances(self, type_conformances: list[TypeConformance]):
        """
        Specify Slang type conformances to use when compiling the function.
        """
        return FunctionNodeTypeConformances(self, type_conformances)

    @property
    def bwds(self):
        """
        Return a new function object that represents the backwards deriviative of the current function.
        """
        return FunctionNodeBwds(self)

    def return_type(self, return_type: Union[type, str]):
        """
        Explicitly specify the desired return type from the function.
        """
        if isinstance(return_type, str):
            if return_type == "numpy":
                import numpy as np

                return_type = np.ndarray
            elif return_type == "tensor":
                from slangpy.types import Tensor

                return_type = Tensor
            else:
                raise ValueError(f"Unknown return type '{return_type}'")
        return FunctionNodeReturnType(self, return_type)

    def thread_group_size(self, thread_group_size: uint3):
        """
        Override the default thread group size for the function. Currently only used for
        raw dispatch.
        """
        return FunctionNodeThreadGroupSize(self, thread_group_size)

    def as_func(self) -> "FunctionNode":
        """
        Typing helper to cast the function to a function (i.e. a no-op)
        """
        return self

    def as_struct(self) -> "Struct":
        """
        Typing helper to detect attempting to treat a function as a struct.
        """
        raise ValueError("Cannot convert a function to a struct")

    def debug_build_call_data(self, *args: Any, **kwargs: Any):
        """
        Debug helper to build call data without dispatching the kernel.
        """
        return cast(
            "CallData",
            self._native_build_call_data(self.module.call_data_cache, *args, **kwargs),
        )

    def call(self, *args: Any, **kwargs: Any) -> Any:
        """
        Call the function with a given set of arguments. This will generate and compile
        a new kernel if need be, then immediately dispatch it and return any results.
        """
        # Handle result type override (e.g. for numpy) by checking
        # for override, and if found, deleting the _result arg and
        # calling the function with the override type.
        resval = kwargs.get("_result", None)
        if isinstance(resval, (type, str)):
            del kwargs["_result"]
            return self.return_type(resval).call(*args, **kwargs)
        else:
            try:
                return self._native_call(self.module.call_data_cache, *args, **kwargs)
            except ValueError as e:
                # If runtime returned useful information, reformat it and raise a new exception
                # Otherwise just throw the original.
                if (
                    len(e.args) != 1
                    or not isinstance(e.args[0], dict)
                    or not "message" in e.args[0]
                    or not "source" in e.args[0]
                    or not "context" in e.args[0]
                ):
                    raise
                from slangpy.bindings.boundvariableruntime import (
                    BoundCallRuntime,
                    BoundVariableRuntime,
                )
                from slangpy.core.native import NativeCallData
                from slangpy.core.logging import bound_runtime_call_table

                msg: str = e.args[0]["message"]
                source: BoundVariableRuntime = e.args[0]["source"]
                context: NativeCallData = e.args[0]["context"]
                runtime = cast(BoundCallRuntime, context.runtime)
                msg += (
                    "\n\n"
                    + bound_runtime_call_table(runtime, source)
                    + "\n\nFor help and support: https://khr.io/slangdiscord"
                )
                raise ValueError(msg) from e

    def append_to(self, command_buffer: CommandEncoder, *args: Any, **kwargs: Any):
        """
        Append the function to a command buffer without dispatching it. As with calling,
        this will generate and compile a new kernel if need be. However the dispatch
        is just added to the command list and no results are returned.
        """
        self._native_append_to(self.module.call_data_cache, command_buffer, *args, **kwargs)

    def dispatch(
        self,
        thread_count: uint3,
        vars: dict[str, Any] = {},
        command_buffer: Optional[CommandEncoder] = None,
        **kwargs: Any,
    ) -> None:
        """
        Perform a raw dispatch, bypassing the majority of SlangPy's typing/code gen logic. This is
        useful if you just want to explicitly call an existing kernel, or treat a slang function
        as a kernel entry point directly.
        """
        if ENABLE_CALLDATA_CACHE:
            if self.slangpy_signature == "":
                build_info = self.calc_build_info()
                lines = []
                if build_info.this_type is not None:
                    lines.append(f"{build_info.this_type.full_name}::{self.name}")
                else:
                    lines.append(build_info.name)
                lines.append(str(build_info.options))
                lines.append(str(build_info.map_args))
                lines.append(str(build_info.map_kwargs))
                lines.append(str(build_info.type_conformances))
                lines.append(str(build_info.call_mode))
                lines.append(str(build_info.return_type))
                lines.append(str(build_info.constants))
                lines.append(str(build_info.thread_group_size))
                lines.append(str(build_info.call_group_shape))
                self.slangpy_signature = "\n".join(lines)

            builder = SignatureBuilder()
            self.module.call_data_cache.get_args_signature(builder, self, **kwargs)
            sig = builder.str

            if sig in self.module.dispatch_data_cache:
                dispatch_data = self.module.dispatch_data_cache[sig]
                if dispatch_data.device != self.module.device:
                    raise NameError("Cached CallData is linked to wrong device")
            else:
                from slangpy.core.dispatchdata import DispatchData

                dispatch_data = DispatchData(self, **kwargs)
                self.module.dispatch_data_cache[sig] = dispatch_data
        else:
            from slangpy.core.dispatchdata import DispatchData

            dispatch_data = DispatchData(self, **kwargs)

        opts = NativeCallRuntimeOptions()
        self.gather_runtime_options(opts)
        dispatch_data.dispatch(opts, thread_count, vars, command_buffer, **kwargs)

    def calc_build_info(self):
        info = FunctionBuildInfo()
        self._populate_build_info_recurse(info)
        return info

    def _populate_build_info_recurse(self, info: FunctionBuildInfo):
        if self._native_parent is not None:
            self.parent._populate_build_info_recurse(info)
        self._populate_build_info(info)

    def _populate_build_info(self, info: FunctionBuildInfo):
        pass

    def _handle_error(self, e: ValueError, calldata: Optional["CallData"]):
        if len(e.args) != 1 or not isinstance(e.args[0], dict):
            raise e
        if not "message" in e.args[0] or not "source" in e.args[0]:
            raise e
        msg = e.args[0]["message"]
        source = e.args[0]["source"]
        raise ValueError(f"Exception dispatching kernel: {msg}\n.")

    def __call__(self, *args: Any, **kwargs: Any):
        """
        Call operator, maps to `call` method.
        """
        return self.call(*args, **kwargs)

    def generate_call_data(self, args: Any, kwargs: Any):
        from .calldata import CallData

        return CallData(self, *args, **kwargs)

    def call_group_shape(self, call_group_shape: Shape):
        """
        Specify the call group shape for the function. This determines how the computation
        is divided into call groups. The shape can be N-dimensional.
        """
        return FunctionNodeCallGroupShape(self, call_group_shape)


class FunctionNodeBind(FunctionNode):
    def __init__(self, parent: NativeFunctionNode, this: IThis) -> None:
        super().__init__(parent, FunctionNodeType.this, this)

    @property
    def this(self):
        return cast(IThis, self._native_data)


class FunctionNodeMap(FunctionNode):
    def __init__(
        self,
        parent: NativeFunctionNode,
        map_args: tuple[Any],
        map_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(parent, FunctionNodeType.kernelgen, (map_args, map_kwargs))
        self.slangpy_signature = str((map_args, map_kwargs))

    @property
    def mapping(self):
        return cast(tuple[tuple[Any, ...], dict[str, Any]], self._native_data)

    @property
    def args(self):
        return self.mapping[0]

    @property
    def kwargs(self):
        return self.mapping[1]

    def _populate_build_info(self, info: FunctionBuildInfo):
        info.map_args = self.args
        info.map_kwargs = self.kwargs


class FunctionNodeSet(FunctionNode):
    def __init__(self, parent: NativeFunctionNode, value: Any) -> None:
        super().__init__(parent, FunctionNodeType.uniforms, value)

    @property
    def uniforms(self):
        return self._native_data


class FunctionNodeConstants(FunctionNode):
    def __init__(self, parent: NativeFunctionNode, constants: dict[str, Any]) -> None:
        super().__init__(parent, FunctionNodeType.kernelgen, constants)
        self.slangpy_signature = str(constants)

    @property
    def constants(self):
        return cast(dict[str, Any], self._native_data)

    def _populate_build_info(self, info: FunctionBuildInfo):
        info.constants.update(self.constants)


class FunctionNodeTypeConformances(FunctionNode):
    def __init__(
        self, parent: NativeFunctionNode, type_conformances: list[TypeConformance]
    ) -> None:
        super().__init__(parent, FunctionNodeType.kernelgen, type_conformances)
        self.slangpy_signature = str(type_conformances)

    @property
    def type_conformances(self):
        return cast(list[TypeConformance], self._native_data)

    def _populate_build_info(self, info: FunctionBuildInfo):
        info.type_conformances.extend(self.type_conformances)


class FunctionNodeBwds(FunctionNode):
    def __init__(self, parent: NativeFunctionNode) -> None:
        super().__init__(parent, FunctionNodeType.kernelgen, None)
        self.slangpy_signature = "bwds"

    def _populate_build_info(self, info: FunctionBuildInfo):
        info.call_mode = CallMode.bwds


class FunctionNodeReturnType(FunctionNode):
    def __init__(self, parent: NativeFunctionNode, return_type: Union[type, str]) -> None:
        super().__init__(parent, FunctionNodeType.kernelgen, return_type)
        self.slangpy_signature = str(return_type)

    @property
    def return_type(self):
        return cast(Union[type, str], self._native_data)

    def _populate_build_info(self, info: FunctionBuildInfo):
        info.return_type = self.return_type


class FunctionNodeThreadGroupSize(FunctionNode):
    def __init__(self, parent: NativeFunctionNode, thread_group_size: uint3) -> None:
        super().__init__(parent, FunctionNodeType.kernelgen, thread_group_size)
        self.slangpy_signature = str(thread_group_size)

    @property
    def thread_group_size(self):
        return cast(uint3, self._native_data)

    def _populate_build_info(self, info: FunctionBuildInfo):
        info.thread_group_size = self.thread_group_size


class FunctionNodeLogger(FunctionNode):
    def __init__(self, parent: NativeFunctionNode, logger: Logger) -> None:
        super().__init__(parent, FunctionNodeType.kernelgen, logger)
        self.slangpy_signature = "logger_" + str(id(logger))

    @property
    def logger(self):
        return cast(Logger, self._native_data)

    def _populate_build_info(self, info: FunctionBuildInfo):
        info.logger = self.logger


class FunctionNodeCallGroupShape(FunctionNode):
    def __init__(self, parent: NativeFunctionNode, call_group_shape: Shape) -> None:
        super().__init__(parent, FunctionNodeType.kernelgen, call_group_shape)
        self.slangpy_signature = str(call_group_shape)

    @property
    def call_group_shape(self):
        return cast(Shape, self._native_data)

    def _populate_build_info(self, info: FunctionBuildInfo):
        info.call_group_shape = self.call_group_shape


class Function(FunctionNode):
    def __init__(
        self,
        module: "Module",
        func: Union[str, SlangFunction],
        struct: Optional["Struct"] = None,
        options: dict[str, Any] = {},
    ) -> None:
        super().__init__(None, FunctionNodeType.kernelgen, None)

        self._module = module

        if isinstance(func, str):
            if struct is None:
                sf = module.layout.find_function_by_name(func)
            else:
                sf = module.layout.find_function_by_name_in_type(struct.struct, func)
            if sf is None:
                raise ValueError(f"Function '{func}' not found")
            func = sf

        # Track fully specialized name
        self._name = func.full_name
        # Store function reflection
        self._slang_func = func

        # Store type parent name if found
        if struct is not None:
            self._this_type = struct.struct
        else:
            self._this_type = None

        # Calc hash of input options for signature
        self._options = options.copy()
        if not "implicit_element_casts" in self._options:
            self._options["implicit_element_casts"] = True
        if not "implicit_tensor_casts" in self._options:
            self._options["implicit_tensor_casts"] = True
        if not "strict_broadcasting" in self._options:
            self._options["strict_broadcasting"] = True

        # Generate signature for hashing
        lines = []
        if self._this_type is not None:
            lines.append(f"{self._this_type.full_name}::{self.name}")
        else:
            lines.append(self.name)
        lines.append(str(self._options))
        self.slangpy_signature = "\n".join(lines)

    def _populate_build_info(self, info: FunctionBuildInfo):
        info.name = self.name
        info.module = self.module
        info.options.update(self._options)
        info.function = self._slang_func
        info.this_type = self._this_type
