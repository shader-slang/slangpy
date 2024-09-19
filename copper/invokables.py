from __future__ import annotations

from .reflection import Modifier, VoidType, SlangFunc
from .reflection import is_flattenable, is_empty_type
from .marshalling import Variable, wrap
from .marshalling.struct import Struct
from .codegen import SlangCodeGen
from .util import broadcast_shapes

from .layers import Program, Kernel, compute_layer, tensor_ref

import math
import logging

from collections import OrderedDict
from typing import Any, Optional

kernel_template = """
import Plugins.NeuralMaterialOptimizer.slangtorch;

// Link time specialization currently broken with reflection...
//extern static const int kMaxBatchDim = {max_batch_dim};

{definitions}

struct ShaderData
{{
    // Metadata
    BatchSize batchSize;
    ArgReader args;
}};
ConstantBuffer<ShaderData> data;

void trampoline(BatchIndex idx)
{
    var args = data.args.read(idx);
    {function_call}
    data.args.write(args);
}

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{{
    if (dispatchThreadId.x >= data.batchSize.flattened)
        return;
    let batchIdx = BatchIndex(data.batchSize, dispatchThreadId.x);

    trampoline(batchIdx);
}}
"""


class SpecializedSlangFunc:
    def __init__(
        self,
        ref_prog: Program,
        root: InvokableSlangFunc,
        wrappers: tuple[Variable, ...],
        have_return: bool,
    ):
        super().__init__()
        self.ref_prog = ref_prog
        self.root = root
        self.wrappers = wrappers
        self.have_return = have_return
        self.kernel: Optional[Kernel] = None
        self.backward_cache = []
        self.buffer_cache = {}
        self.is_method = False
        self.max_batch_dim = 4  # TODO

    def create_kernel(self):
        definition_gen = SlangCodeGen()
        definition_gen.begin_block('struct FunctionArguments {')
        for p in self.root.func_params:
            definition_gen.emit(f"{p.type} {p.name}")
        definition_gen.end_block()
        definition_gen.begin_block('struct ArgReader : Variable<FunctionArguments> {')
        for param, wrapper in zip(self.root.func_params, self.wrappers):
            wrapper.declare(definition_gen, param.name)

        definition_gen.begin_block('FunctionArguments read(BatchIndex idx) {')
        definition_gen.emit('FunctionAguments result;')
        for param, wrapper in zip(self.root.func_params, self.wrappers):
            if param.is_in():
                definition_gen.emit(f"result.{param.name} = {param.name}.read(idx);")
        definition_gen.emit('return result;')
        definition_gen.end_block()
        definition_gen.begin_block('void write(BatchIndex idx, FunctionArguments args) {')
        for param, wrapper in zip(self.root.func_params, self.wrappers):
            if param.is_out():
                definition_gen.emit(f"{param.name}.write(idx, args.{param.name});")
        definition_gen.end_block()

        call_base = self.root.func_name
        call_args = ["args." + param.name for param in self.root.func_params]

        # Because function return values require much the same handling as out parameters,
        # we add the function return value as an out parameter at the end of the parameter
        # list when we create the specialized function. The only place where the distinction
        # between out parameter and return value matters is when we call the function
        # in the slang code. We handle this here.
        if self.have_return:
            call_base = f"{call_args[-1]} = {call_base}"
            call_args.pop()

        call_expression = f'{call_base}({", ".join(call_args)});'

        wrapper_code = kernel_template.format(
            max_batch_dim=self.max_batch_dim,
            definitions=definition_gen.code(),
            function_call=call_expression,
        )

        if logging.root.isEnabledFor(logging.DEBUG):
            logging.debug(f"Shader code for {self.root.func_name}:")
            logging.debug(wrapper_code)

        logging.debug(f"Compiling kernel for {self.root.func_name}...")
        prog = compute_layer().extend_program(
            self.ref_prog, "KfWrapper", wrapper_code, entry_point="main"
        )
        self.kernel = compute_layer().create_kernel(prog)
        logging.debug(f"Compilation complete")

    def __call__(self, values: list[Any]):
        self.call(values)

    def call(self, values: list[Any]):
        sizes = [var.batch_size(value) for var, value in zip(self.wrappers, values)]
        batch_size = broadcast_shapes(sizes)

        if batch_size is None:
            lines = []
            lines.append(
                f"Could not invoke slang function {self.root.func_name} because "
                "the batch sizes could not be broadcast together.\n"
                "Below are the inferred batch sizes for each parameter:"
            )
            for param, value, shape in zip(self.root.func_params, values, sizes):
                lines.append(f"    {param.type} {param.name}: "
                             f"{type(value).__name__} -> {list(shape)}")

            raise ValueError("\n".join(lines))

        # TODO ...
        if self.max_batch_dim is None:
            # self.max_batch_dim = len(batch_size)
            self.max_batch_dim = 4

        batch_count = math.prod(batch_size)

        if self.kernel is None:
            self.create_kernel()
            assert self.kernel is not None

        block = self.kernel.rootvar()["data"]
        block["batchSize"]["flattened"] = batch_count
        block["batchSize"]["dims"] = [1] * (
            self.max_batch_dim - len(batch_size)
        ) + list(batch_size)

        for param, var, value in zip(self.root.func_params, self.wrappers, values):
            var.assign(block['args'][param.name], value, batch_size)

        self.kernel.dispatch(batch_count)


class InvokableSlangFunc:
    def __init__(self, prog: Program, func: str | SlangFunc):
        super().__init__()

        self.ref_prog = prog

        if isinstance(func, str):
            self.func_name = func
            refl_func = prog.find_function(self.func_name)
            if not refl_func:
                raise ValueError(f"Can't find function {self.func_name}")
            self.func = refl_func
        else:
            self.func = func
            self.func_name = func.name

        self.func_params = self.func.params[:]
        if is_empty_type(self.func.return_type):
            self.have_return = False
        else:
            # if not StructReturnValue.is_compatible(self.func.return_type):
            if not is_flattenable(self.func.return_type):
                raise ValueError(
                    f"Return type {self.func.return_type} of "
                    f"function {self.func_name} is not compatible with tensors"
                )

            self.func_params.append(self.func.get_return_param())
            self.have_return = True

        self.specializations: list[tuple[tuple[Variable, ...], SpecializedSlangFunc]] = []
        self.d_func = None

    def differentiate(self) -> InvokableSlangFunc:
        if self.d_func is None:
            self.d_func = InvokableSlangFunc(self.ref_prog, self.func.differentiate())
            self.d_func.func_name = f"bwd_diff({self.func_name})"

        return self.d_func

    def try_populate_params(
        self, positional_args: tuple[Any, ...], keyword_args: dict[str, Any]
    ):
        param_map = OrderedDict((param.name, None) for param in self.func.params)

        if len(positional_args) > len(self.func.params):
            raise ValueError(
                f"Too many arguments for slang function {self.func_name}: "
                f"Received {len(positional_args)}, expected {len(self.func.params)}"
            )

        for i, arg in enumerate(positional_args):
            param_map[self.func.params[i].name] = arg

        for k, v in keyword_args.items():
            if k not in param_map:
                raise ValueError(
                    f"Invalid argument '{k}' to slang function '{self.func_name}'. "
                    f"Valid arguments are: {', '.join(param_map.keys())}"
                )
            if param_map[k] is not None:
                raise ValueError(
                    f"Argument {k} for slang function {self.func_name} is set multiple times"
                )
            param_map[k] = v

        missing_args = [k for k, v in param_map.items() if v is None]
        if missing_args:
            raise ValueError(
                f"Function '{self.func_name}' is missing required argument(s) "
                f"{', '.join(missing_args)}"
            )

        return param_map

    def specialize(self, wrappers: tuple[Variable, ...]) -> SpecializedSlangFunc:
        specialized_fun = None
        logging.debug(f"Specializing {self.func_name}")
        for key, fun in self.specializations:
            if key == wrappers:
                specialized_fun = fun
                break

        if not specialized_fun:
            logging.debug(f"    Creating new specialization")

            specialized_fun = SpecializedSlangFunc(
                self.ref_prog,
                self,
                wrappers,
                self.have_return
            )
            self.specializations.append((wrappers, specialized_fun))

        return specialized_fun

    def __call__(self, *args: Any, **kwargs: Any):
        return self.call(*args, **kwargs)

    def call(self, *args: Any, **kwargs: Any):
        input_map = self.try_populate_params(args, kwargs)
        inputs: list[Any] = list(input_map.values())

        if self.have_return:
            # inputs.append(tensor_ref() if is_flattenable(self.func.return_type) else StructReturnValue())
            inputs.append(tensor_ref())

        # TODO: Begin hook
        wrapped = [wrap(p.type, v, p.modifiers) for p, v in zip(self.func_params, inputs)]
        vars, values = zip(*wrapped)
        self.specialize(vars)(values)
        # TODO: End hook

        if self.have_return:
            return inputs[-1]


class InvokableSlangMethod:
    def __init__(self, method: SlangFunc, self_value: Struct, prog: Program):
        super().__init__()
        self.method = method
        self.self_value = self_value
        self.prog = prog
        self.invokable = None

    def create_invokable(self):
        struct = self.self_value._type
        return_type = self.method.return_type

        func_name = struct.name.base + "__" + self.method.name
        params = [repr(p) for p in self.method.params]
        self_modifier = (
            "inout" if Modifier.Mutating in self.method.func_modifiers else "in"
        )
        if Modifier.NoDiffThis in self.method.func_modifiers:
            self_modifier += " no_diff"
        self_param = f"{self_modifier} {struct} self"
        param_definition = ", ".join([self_param] + params)
        param_use = ", ".join(p.name for p in self.method.params)

        prefix = "return " if not isinstance(return_type, VoidType) else ""
        code = (
            f"{return_type} {func_name}({param_definition}) {{\n"
            f"    {prefix}self.{self.method.name}({param_use});\n"
            "}\n"
        )

        if Modifier.BackwardDifferentiable in self.method.func_modifiers:
            code = "[BackwardDifferentiable]\n" + code

        method_prog = compute_layer().extend_program(self.prog, "MethodWrapper", code)

        self.invokable = InvokableSlangFunc(method_prog, func_name)

    def get_invokable(self) -> InvokableSlangFunc:
        if self.invokable is None:
            self.create_invokable()
            assert self.invokable is not None
        return self.invokable

    def __call__(self, *args: Any, **kwargs: Any):
        return self.call(*args, **kwargs)

    def call(self, *args: Any, **kwargs: Any):
        return self.get_invokable().call(self.self_value, *args, **kwargs)
