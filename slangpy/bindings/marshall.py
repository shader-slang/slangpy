# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from typing import TYPE_CHECKING, Any

from slangpy.core.native import CallMode, CallDataMode, NativeMarshall

from slangpy.bindings.codegen import CodeGenBlock

if TYPE_CHECKING:
    from slangpy import SlangModule, ShaderObject, Device
    from slangpy.bindings.boundvariable import BoundVariable
    from slangpy.reflection import SlangProgramLayout, SlangType


class BindContext:
    """
    Contextual information passed around during kernel generation process.
    """

    def __init__(
        self,
        layout: "SlangProgramLayout",
        call_mode: CallMode,
        device_module: "SlangModule",
        options: dict[str, Any],
        call_data_mode: CallDataMode,
    ):
        super().__init__()

        #: The layout of the program being generated.
        self.layout = layout

        #: Call dimensionality (-1 until calculated).
        self.call_dimensionality = -1

        #: Call mode (prim/bwds/fwds).
        self.call_mode = call_mode

        #: Call data mode (global_data/entry_point).
        self.call_data_mode = call_data_mode

        #: SGL module.
        self.device_module = device_module

        #: Kernel gen options.
        self.options = options

    @property
    def device(self) -> "Device":
        """
        The device this context is bound to.
        """
        return self.device_module.session.device


class ReturnContext:
    """
    Internal structure used to store information about return type of a function during generation.
    """

    def __init__(self, slang_type: "SlangType", bind_context: BindContext):
        super().__init__()

        #: The slang type to return.
        self.slang_type = slang_type

        #: Cached bind context.
        self.bind_context = bind_context


class Marshall(NativeMarshall):
    """
    Base class for a type marshall that describes how to pass a given type to/from a
    SlangPy kernel. When a kernel is generated, a marshall is instantiated for each
    Python value. Future calls to the kernel verify type signatures match and then
    re-use the existing marshalls.
    """

    def __init__(self, layout: "SlangProgramLayout"):
        super().__init__()

        #: The slang type the python value maps to. Should be set inside __init__
        self.slang_type: "SlangType"

    @property
    def has_derivative(self) -> bool:
        """
        Does value have a derivative. Default: False
        """
        return super().has_derivative

    @property
    def is_writable(self) -> bool:
        """
        Is value writable. Default: False
        """
        return super().is_writable

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: "BoundVariable"):
        """
        Generate the code for the uniforms that will represent this value in the kernel.
        Raises exception if not overriden.
        """
        return super().gen_calldata(cgb, context, binding)

    def reduce_type(self, context: BindContext, dimensions: int) -> "SlangType":
        """
        Get the slang type for this variable when a given number of dimensions
        are removed. i.e. if the variable is a matrix, reduce_type(1) would
        return a vector, and reduce_type(2) would return a scalar. Raises
        exception if needed and not overriden.
        """
        res = super().reduce_type(context, dimensions)
        return res  # type: ignore

    def resolve_type(self, context: BindContext, bound_type: "SlangType") -> "SlangType":
        """
        Return the slang type for this variable when passed to a parameter
        of the given type. Default behaviour simply attempts to pass its own type,
        but more complex behaviour can be added to support implicit casts. Default to just
        casting to itself (i.e. no implicit cast)
        """
        res = super().resolve_type(context, bound_type)
        return res  # type: ignore

    def resolve_dimensionality(
        self,
        context: BindContext,
        binding: "BoundVariable",
        vector_target_type: "SlangType",
    ):
        """
        Calculate the call dimensionality when this value is passed as a given type. For example,
        a 3D buffer passed to a scalar would return 3, but a 3D buffer passed to a 3D buffer would
        return 0.

        Default implementation simply returns the difference between the dimensionality of this
        type and the target type.
        """
        return super().resolve_dimensionality(context, binding, vector_target_type)

    def build_shader_object(self, context: BindContext, data: Any) -> "ShaderObject":
        """
        Build a shader object from the given data. This is called when attempting to finalize
        a value before passing it as a read only shader object.
        """
        return super().build_shader_object(data)
