# SPDX-License-Identifier: Apache-2.0
from copy import copy
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol, Union

from slangpy.core.native import (CallMode, NativeCallRuntimeOptions, NativeObject,
                                 hash_signature)

import slangpy.reflection as kfr
from slangpy.backend import (CommandBuffer, FunctionReflection,
                             TypeConformance, TypeReflection, uint3)
from slangpy.bindings.typeregistry import PYTHON_SIGNATURES

if TYPE_CHECKING:
    from slangpy.core.calldata import CallData
    from slangpy.core.module import Module
    from slangpy.core.struct import Struct

ENABLE_CALLDATA_CACHE = True


TCallHook = Callable[['Function'], None]


def _cache_value_to_id(val: Any) -> str:
    cb = PYTHON_SIGNATURES.get(type(val))
    if cb is None:
        return ""
    else:
        return cb(val)


class IThis(Protocol):
    def get_this(self) -> Any:
        ...

    def update_this(self, value: Any) -> None:
        ...


class Function(NativeObject):
    """
    Callable class that represents a Slang function in a loaded module. Typically created
    by calling `module.function_name` or `mystruct.function_name` on a loaded module/struct.
    """

    def __init__(self) -> None:
        super().__init__()
        self.module: 'Module'
        self.type_reflection: Optional['TypeReflection']
        self.reflections: list['FunctionReflection']
        self._name: str

        # Static options that affect compilation, and thus the signature
        self._map_args: Optional[tuple[Any]] = None
        self._map_kwargs: Optional[dict[str, Any]] = None
        self._options: Optional[dict[str, Any]] = None
        self._type_conformances: Optional[list[TypeConformance]] = None
        self._mode = CallMode.prim
        self._return_type: Optional[type] = None
        self._constants: Optional[dict[str, Any]] = None
        self._thread_group_size: Optional[uint3] = None

        # Runtime options that affect dispatch only
        self.this: Optional[IThis] = None
        self.uniforms: Optional[list[Union[Callable[[
            'CallData'], Any], dict[str, Any]]]] = None

    def torch(self):
        """
        Returns a pytorch wrapper around this function
        """
        import slangpy.torchintegration as spytorch
        if spytorch.TORCH_ENABLED:
            return spytorch.TorchFunction(self)
        else:
            raise RuntimeError("Pytorch integration is not enabled")

    def _copy(self) -> 'Function':
        res = Function()
        res.module = self.module
        res.type_reflection = self.type_reflection
        res.reflections = self.reflections
        res._name = self._name
        res._map_args = self._map_args
        res._map_kwargs = self._map_kwargs
        res._options = self._options
        res._type_conformances = self._type_conformances
        res._mode = self._mode
        res._return_type = self._return_type
        res._constants = self._constants
        res._thread_group_size = self._thread_group_size
        res.this = self.this
        res.uniforms = self.uniforms
        res.slangpy_signature = ''
        return res

    def attach(self, module: 'Module', func: Union[str, kfr.SlangFunction, list[FunctionReflection]], struct: Optional['Struct'] = None, options: dict[str, Any] = {}) -> None:
        """
        Links a function to its parent module or struct. Typically only called internally by SlangPy.
        """

        self.module = module

        if isinstance(func, str):
            if struct is None:
                sf = module.layout.find_function_by_name(func)
            else:
                sf = module.layout.find_function_by_name_in_type(struct.struct, func)
            if sf is None:
                raise ValueError(f"Function '{func}' not found")
            func = sf

        if isinstance(func, kfr.SlangFunction):
            func_reflections = [func.reflection]
            # Track fully specialized name where available
            self._name = func.full_name
        else:
            func_reflections = func
            self._name = func[0].name

        # Store function reflections (should normally be 1 unless forced to do AST based search)
        self.reflections = func_reflections

        # Store type parent name if found
        if struct is not None:
            self.type_reflection = struct.struct.type_reflection
        else:
            self.type_reflection = None

        # Calc hash of input options for signature
        self._options = options.copy()
        if not 'implicit_element_casts' in self._options:
            self._options['implicit_element_casts'] = True
        if not 'implicit_tensor_casts' in self._options:
            self._options['implicit_tensor_casts'] = True
        if not 'strict_broadcasting' in self._options:
            self._options['strict_broadcasting'] = True

        # Generate signature for hashing
        # type_parent = self.type_reflection.full_name if self.type_reflection is not None else None
        # self.slangpy_signature = f"[{type_parent or ''}::{self.name},{options_hash}]"

    def bind(self, this: IThis) -> 'Function':
        """
        Bind a `this` object to the function. Typically
        this is called automatically when calling a function on a struct.
        """
        res = self._copy()
        res.this = this
        return res

    def map(self, *args: Any, **kwargs: Any):
        """
        Apply dimension or type mapping to all or some of the arguments.

        myfunc.map((1,)(0,))(arg1, arg2) # Map arg1 to dimension 1, arg2 to dimension 0

        myfunc.map(module.Foo, module.Bar)(arg1, arg2) # Cast arg1 to Foo, arg2 to Bar
        """
        res = self._copy()
        res._map_args = args
        res._map_kwargs = kwargs
        return res

    def set(self, *args: Any, **kwargs: Any):
        """
        Specify additional uniform values that should be set whenever the function's kernel
        is dispatched. Useful for setting constants or other values that are not passed as arguments.
        """

        res = self._copy()

        if len(args) > 0 and len(kwargs) > 0:
            raise ValueError(
                "Set accepts either positional or keyword arguments, not both"
            )
        if len(args) > 1:
            raise ValueError(
                "Set accepts only one positional argument (a dictionary or callback)"
            )

        if len(kwargs) > 0:
            res._add_uniform_values(kwargs)
        elif len(args) > 0:
            if callable(args[0]):
                res._add_uniform_callback(args[0])
            elif isinstance(args[0], dict):
                res._add_uniform_values(args[0])
            else:
                raise ValueError(
                    "Set requires a dictionary or callback as a single positional argument"
                )
        else:
            raise ValueError("Set requires at least one argument")

        return res

    def _add_uniform_values(self, uniform_values: dict[str, Any]):
        if self.uniforms is None:
            self.uniforms = [uniform_values]
        else:
            self.uniforms = copy(self.uniforms)
            self.uniforms.append(uniform_values)

    def _add_uniform_callback(self, uniform_callback: Callable[['CallData'], Any]):
        if self.uniforms is None:
            self.uniforms = [uniform_callback]
        else:
            self.uniforms = copy(self.uniforms)
            self.uniforms.append(uniform_callback)

    def constants(self, constants: dict[str, Any]):
        """
        Specify link time constants that should be set when the function is compiled. These are
        the most optimal way of specifying unchanging data, however note that changing a constant
        will result in the function being recompiled.
        """
        res = self._copy()
        if res._constants is None:
            res._constants = constants
        else:
            res._constants = copy(res._constants)
            res._constants.update(constants)
        return res

    def type_conformances(self, type_conformances: list[TypeConformance]):
        """
        Specify Slang type conformances to use when compiling the function.
        """
        res = self._copy()
        if res._type_conformances is None:
            res._type_conformances = type_conformances
        else:
            res._type_conformances = copy(res._type_conformances)
            res._type_conformances.extend(type_conformances)
        return res

    @property
    def bwds(self):
        """
        Return a new function object that represents the backwards deriviative of the current function.
        """
        res = self._copy()
        res._mode = CallMode.bwds
        return res

    def return_type(self, return_type: Union[type, str]):
        """
        Explicitly specify the desired return type from the function.
        """
        res = self._copy()
        if isinstance(return_type, str):
            if return_type == 'numpy':
                import numpy as np
                return_type = np.ndarray
            elif return_type == 'tensor':
                from slangpy.types import Tensor
                return_type = Tensor
            else:
                raise ValueError(f"Unknown return type '{return_type}'")
        res._return_type = return_type
        return res

    def thread_group_size(self, thread_group_size: uint3):
        """
        Override the default thread group size for the function. Currently only used for
        raw dispatch.
        """
        res = self._copy()
        res._thread_group_size = thread_group_size
        return res

    @property
    def name(self):
        """
        Get the name of the function.
        """
        return self._name

    def as_func(self) -> 'Function':
        """
        Typing helper to cast the function to a function (i.e. a no-op)
        """
        return self

    def as_struct(self) -> 'Struct':
        """
        Typing helper to detect attempting to treat a function as a struct.
        """
        raise ValueError("Cannot convert a function to a struct")

    def call(self, *args: Any, **kwargs: Any) -> Any:
        """
        Call the function with a given set of arguments. This will generate and compile
        a new kernel if need be, then immediately dispatch it and return any results.
        """

        # Handle result type override (e.g. for numpy) by checking
        # for override, and if found, deleting the _result arg and
        # calling the function with the override type.
        resval = kwargs.get('_result', None)
        if isinstance(resval, (type, str)):
            del kwargs['_result']
            return self.return_type(resval).call(*args, **kwargs)

        calldata: Optional['CallData'] = None
        try:
            if self.this:
                args = (self.this,)+args
            calldata = self._build_call_data(*args, **kwargs)
            opts = NativeCallRuntimeOptions()
            if self.uniforms is not None:
                opts.uniforms = self.uniforms  # type: ignore (can't work out this type)
            res = calldata.call(opts, *args, **kwargs)
            return res
        except ValueError as e:
            self._handle_error(e, calldata)

    def append_to(self, command_buffer: CommandBuffer, *args: Any, **kwargs: Any):
        """
        Append the function to a command buffer without dispatching it. As with calling,
        this will generate and compile a new kernel if need be. However the dispatch
        is just added to the command list and no results are returned.
        """
        calldata: Optional['CallData'] = None
        try:
            if self.this:
                args = (self.this,)+args
            calldata = self._build_call_data(*args, **kwargs)
            opts = NativeCallRuntimeOptions()
            if self.uniforms is not None:
                opts.uniforms = self.uniforms  # type: ignore (can't work out this type)
            return calldata.append_to(opts, command_buffer, *args, **kwargs)
        except ValueError as e:
            self._handle_error(e, calldata)

    def dispatch(self, thread_count: uint3, vars: dict[str, Any] = {}, command_buffer: CommandBuffer | None = None, **kwargs: Any) -> None:
        """
        Perform a raw dispatch, bypassing the majority of SlangPy's typing/code gen logic. This is
        useful if you just want to explicitly call an existing kernel, or treat a slang function
        as a kernel entry point directly.
        """
        if ENABLE_CALLDATA_CACHE:
            if self.slangpy_signature == '':
                lines = []
                if self.type_reflection is not None:
                    lines.append(f"{self.type_reflection.full_name}::{self.name}")
                else:
                    lines.append(self.name)
                lines.append(str(self._options))
                lines.append(str(self._map_args))
                lines.append(str(self._map_kwargs))
                lines.append(str(self._type_conformances))
                lines.append(str(self._mode))
                lines.append(str(self._return_type))
                lines.append(str(self._constants))
                lines.append(str(self._thread_group_size))
                self.slangpy_signature = "\n".join(lines)
            sig = hash_signature(
                _cache_value_to_id, self, **kwargs)

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
        if self.uniforms is not None:
            opts.uniforms = self.uniforms  # type: ignore (can't work out this type)
        dispatch_data.dispatch(opts, thread_count, vars, command_buffer, **kwargs)

    def _handle_error(self, e: ValueError, calldata: Optional['CallData']):
        if len(e.args) != 1 or not isinstance(e.args[0], dict):
            raise e
        if not 'message' in e.args[0] or not 'source' in e.args[0]:
            raise e
        msg = e.args[0]['message']
        source = e.args[0]['source']
        raise ValueError(
            f"Exception dispatching kernel: {msg}\n.")

    def debug_build_call_data(self, *args: Any, **kwargs: Any):
        return self._build_call_data(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any):
        """
        Call operator, maps to `call` method.
        """
        return self.call(*args, **kwargs)

    def _build_call_data(self, *args: Any, **kwargs: Any):

        if self.slangpy_signature == '':
            lines = []
            if self.type_reflection is not None:
                lines.append(f"{self.type_reflection.full_name}::{self.name}")
            else:
                lines.append(self.name)
            lines.append(str(self._options))
            lines.append(str(self._map_args))
            lines.append(str(self._map_kwargs))
            lines.append(str(self._type_conformances))
            lines.append(str(self._mode))
            lines.append(str(self._return_type))
            lines.append(str(self._constants))
            self.slangpy_signature = "\n".join(lines)

        sig = hash_signature(
            _cache_value_to_id, self, *args, **kwargs)
        if ENABLE_CALLDATA_CACHE and sig in self.module.call_data_cache:
            cd = self.module.call_data_cache[sig]
            # if cd.device != self.module.device:
            #    raise NameError("Cached CallData is linked to wrong device")
            return cd

        from .calldata import CallData
        res = CallData(self, *args, **kwargs)
        if ENABLE_CALLDATA_CACHE:
            self.module.call_data_cache[sig] = res
        return res
