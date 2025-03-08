# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from ..basetypes import Real

from slangpy import Module, Tensor
from slangpy.core.function import FunctionNode
import sgl

from typing import Optional


class Optimizer:
    def initialize(self, module: Module, parameters: list[Tensor]):
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
        this = self.get_this()
        for param, state, step_func in zip(self.parameters, self.states, self.step_funcs):
            if cmd is None:
                step_func(this, state, param, param.grad)
            else:
                step_func.append_to(cmd, this, state, param, param.grad)

    def get_type_name(self, dtype: Real) -> str:
        raise NotImplementedError()

    def get_this(self):
        raise NotImplementedError
