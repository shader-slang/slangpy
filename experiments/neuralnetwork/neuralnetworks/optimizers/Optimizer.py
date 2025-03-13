# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from ..basetypes import Real

from slangpy import Module, Tensor
from slangpy.core.function import FunctionNode
import sgl

from typing import Optional


class Optimizer:
    """
    This is the base class of all optimizers.

    Creating an optimizer is done in two phases: First, by calling the constructor
    of the optimizer (e.g. AdamOptimizer) and setting its parameters. This is light-weight
    and does not do much work yet.
    Second, by calling .initialize(module, parameters), passing in the slang module containing
    the required slang types and a list of network parameters to optimize. This may perform
    allocation and reflection work.

    .step() performs one optimization step and resets the network gradients.

    For implementers of new optimizers, the following methods should be overridden:
    - get_type_name(dtype) returning the name of a slang type implementing IOptimizer<dtype>
    - get_this(), returning a python type that may be passed to slang (e.g. a dict)
    """

    def __init__(self):
        super().__init__()
        self._initialized = False

    def initialize(self, module: Module, parameters: list[Tensor]):
        """
        Initializes the optimizer from a list of trainable parameters.

        The optimizer must be initialized before it can be used.

        module is a loaded slang module containing the required slang types.

        Parameter tensors don't all have to have the same precision, and it is allowed to use networks
        with e.g. mixed float and half precision parameters.
        """
        self._initialized = True
        self.parameters = parameters
        self.states = []
        self.step_funcs: list[FunctionNode] = []

        for i, param in enumerate(parameters):
            dtype = Real.from_slangtype(param.dtype)
            if dtype is None:
                raise ValueError(f"Unsupported element type '{param.dtype.full_name}' "
                                 f"of parameter {i}: Must be half, float or double")

            type_name = self.get_type_name(dtype)
            optim_type = module.find_struct(type_name)
            if optim_type is None:
                raise ValueError(f"Could not find optimizer type '{type_name}' in slang module '{module.name}'. "
                                 "This could be due to a missing import or a type error. Make sure "
                                 "this is a valid type in the module, e.g. by pasting in the type above "
                                 "and checking for compile errors")

            state_type = module.find_struct(f"{type_name}::State")
            if state_type is None:
                raise ValueError(f"Could not find optimizer state type '{type_name}::State' in slang module "
                                 f"'{module.name}'. Make sure the type {type_name} implements IOptimizer<{dtype.slang()}>")

            step_func = module.find_function_in_struct(optim_type, "step")
            if step_func is None:
                raise ValueError(f"Could not find method '{type_name}::step()' in slang module '{module.name}'. "
                                 f"Make sure the type {type_name} implements IOptimizer<{dtype.slang()}>")

            self.states.append(state_type(param))
            self.step_funcs.append(step_func)

    def step(self, cmd: Optional[sgl.CommandBuffer] = None):
        """
        Performs one step of the optimizer and resets network gradients.

        If cmd is provided, the slang calls are appended to the given command buffer.
        """
        self.check_initialized()

        this = self.get_this()
        for param, state, step_func in zip(self.parameters, self.states, self.step_funcs):
            if cmd is None:
                step_func(this, state, param, param.grad)
            else:
                step_func.append_to(cmd, this, state, param, param.grad)

    def get_type_name(self, dtype: Real) -> str:
        """Returns the name of a slang type implementing IOptimizer<dtype>"""
        raise NotImplementedError()

    def get_this(self):
        """
        Returning a python type that may be passed to slang (e.g. a dict)

        Currently, this type has to be compatible with any optimizer precision.
        This may change in the future.
        """
        raise NotImplementedError()

    def check_initialized(self):
        if not self._initialized:
            raise RuntimeError("Optimizer is uninitialized. Make sure to "
                               "call .initialize() before using the optimizer")
