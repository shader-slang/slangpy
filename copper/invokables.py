from __future__ import annotations

from .types.base import Modifier, VoidType

from .types.interfaces import BatchedType, CodegenType
from .types.func import SlangFunc
from .types.typeutil import is_flattenable, is_empty_type
from .variable import Variable, AssignmentCursor
from .codegen import SlangCodeGen
from .util import broadcast_shapes

from .layers import Program, Kernel, TensorRef, compute_layer, tensor_layer, tensor_ref

import math
import logging

from collections import OrderedDict
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .types.struct import Struct

kernel_template = """
import Plugins.NeuralMaterialOptimizer.slangtorch;

// Link time specialization currently broken with reflection...
//extern static const int kMaxBatchDim = {max_batch_dim};

{definitions}

struct ShaderData
{{
    // Metadata
    BatchSize batchSize;
    {parameter_block}
}};
ConstantBuffer<ShaderData> data;

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{{
    if (dispatchThreadId.x >= data.batchSize.flattened)
        return;
    let batchIdx = BatchIndex(data.batchSize, dispatchThreadId.x);

    var args = {read_args};
    {call_expression}
    {write_results}
}}
"""


class SpecializedSlangFunc:
    def __init__(
        self,
        ref_prog: Program,
        func_name: str,
        root_var: Variable,
        have_return: bool,
        print_enabled: bool,
    ):
        super().__init__()
        self.ref_prog = ref_prog
        self.func_name = func_name
        self.root_var = root_var
        self.have_return = have_return
        self.kernel: Optional[Kernel] = None
        self.backward_cache = []
        self.buffer_cache = {}
        self.is_method = False
        self.max_batch_dim = None
        self.print_enabled = print_enabled

    def create_kernel(self):
        args_name = f"data.{self.root_var.name}"

        definition_gen = SlangCodeGen()
        emitted = set()
        for var in self.root_var.nodes():
            if isinstance(var.type, CodegenType) and var.type not in emitted:
                var.type.define_type(definition_gen)
                emitted.add(var.type)

        param_gen = SlangCodeGen(1)
        param_gen.blank_line()
        param_gen.emit("// Parameters")
        self.root_var.declare(param_gen)

        write_gen = SlangCodeGen(1)
        write_gen.blank_line()
        self.root_var.write(args_name, "args", write_gen)

        call_base = self.func_name
        call_args = ["args." + var.name for var in self.root_var.children.values()]

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
            parameter_block=param_gen.code(),
            read_args=self.root_var.read(args_name),
            call_expression=call_expression,
            write_results=write_gen.code(),
        )

        if logging.root.isEnabledFor(logging.DEBUG):
            logging.debug(f"Shader code for {self.func_name}:")
            logging.debug(wrapper_code)

            logging.debug("Corresponding parameter block:")
            logging.debug(repr(self.root_var))

        logging.debug(f"Compiling kernel for {self.func_name}...")
        prog = compute_layer().extend_program(
            self.ref_prog, "KfWrapper", wrapper_code, entry_point="main"
        )
        self.kernel = compute_layer().create_kernel(prog)
        logging.debug(f"Compilation complete")

    def __call__(self, values: list[Any]):
        self.call(values)

    def call(self, values: list[Any]):
        batched_inputs: list[tuple[Variable, Any, tuple[int, ...]]] = []
        for var, val in zip(self.root_var.nodes(), values):
            if isinstance(var.translator, BatchedType):
                size = var.translator.infer_batch_size(val)
                batched_inputs.append((var, val, size))
        batch_size = broadcast_shapes([size for _, _, size in batched_inputs])

        if logging.root.isEnabledFor(logging.DEBUG):
            logging.debug(f"Derived batch size {batch_size} from batched variables:")
            for var, val, size in batched_inputs:
                prefix = ""
                if isinstance(val, TensorRef):
                    if val.is_empty():
                        prefix = f"Empty tensor -> "
                    else:
                        dtype = val.get_dtype().to_slang_type()
                        prefix = f"tensor.{dtype}{list(val.get_shape())} -> "
                logging.debug(
                    f"    {var.get_root_path_string()} ({var.type.to_slang_type()}): {prefix}{list(size)}"
                )

        if batch_size is None:
            lines = []
            lines.append(
                f"Could not invoke slang function {self.func_name} because "
                "the batch sizes could not be broadcast together.\n"
                "Below are the tensors that were mapped to function inputs "
                "and their inferred batch sizes:"
            )
            for var, value, shape in batched_inputs:
                var_str = (
                    f"    {var.get_root_path_string()} ({var.type.to_slang_type()}): "
                )
                if isinstance(value, TensorRef):
                    shape_str = (
                        f"Tensor shape {list(value.get_shape())} -> batch size {shape}"
                    )
                else:
                    shape_str = f"Batch size {shape}"
                lines.append(var_str + shape_str)

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

        for var, value in zip(self.root_var.nodes(), values):
            if not var.is_leaf:
                continue

            shader_var = block[self.root_var.name]
            for p in var.path():
                shader_var = shader_var[p]

            if isinstance(var.translator, BatchedType):
                value = var.translator.broadcast(value, batch_size)

            if var.is_out:
                assert isinstance(value, TensorRef)

            if isinstance(value, TensorRef):
                shader_var.set_tensor(value, var.is_in, var.is_out)
            elif value is not None:
                shader_var.set(value)

        self.kernel.dispatch(batch_count)


"""
class StructReturnValue(dict[str, Any]):
    def __getattr__(self, name: str):
        if name in self:
            return self.__getitem__(name)
        else:
            raise AttributeError

    def __setattr__(self, name: str, value: Any):
        return self.__setitem__(name, value)

    def __repr__(self):
        def recurse(val: Any):
            if isinstance(val, list):
                return '[' + ', '.join(recurse(v) for v in val) + ']'
            elif isinstance(val, dict):
                return '{' + ', '.join(k + ': ' + recurse(v) for k, v in val.items()) + '}'
            elif tensor_layer().is_tensor(val):
                return 'tensor.' + str(tensor_layer().wrap_tensor(val).get_dtype()) + str(list(val.shape))
            else:
                return repr(val)

        return recurse(self)

    @staticmethod
    def is_compatible(type: SlangType):
        if isinstance(type, StructType):
            return all(StructReturnValue.is_compatible(v) for v in type.members.values())
        return is_flattenable(type)
"""


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
                    f"Return type {self.func.return_type.to_slang_type()} of "
                    f"function {self.func_name} is not compatible with tensors"
                )

            self.func_params.append(self.func.get_return_param())
            self.have_return = True

        self.specializations: list[tuple[str, SpecializedSlangFunc]] = []
        self.d_func = None
        self.print_enabled = False

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

    def specialize(self, root_var: Variable) -> SpecializedSlangFunc:
        translations = [var for var in root_var.leaves() if var.translator is not None]
        instance_key = ",".join(
            f"{var.get_root_path_string()}:{repr(var.translator)}"
            for var in translations
        )
        specialized_fun = None
        logging.debug(f"Specializing {self.func_name} with key {instance_key}")
        for key, fun in self.specializations:
            if key == instance_key:
                specialized_fun = fun
                break

        if not specialized_fun:
            logging.debug(f"    Creating new specialization")

            specialized_fun = SpecializedSlangFunc(
                self.ref_prog,
                self.func_name,
                root_var,
                self.have_return,
                self.print_enabled,
            )
            self.specializations.append((instance_key, specialized_fun))

        return specialized_fun

    def __call__(self, *args: Any, **kwargs: Any):
        return self.call(*args, **kwargs)

    def call(self, *args: Any, **kwargs: Any):
        input_map = self.try_populate_params(args, kwargs)
        inputs: list[Any] = list(input_map.values())

        if self.have_return:
            # inputs.append(tensor_ref() if is_flattenable(self.func.return_type) else StructReturnValue())
            inputs.append(tensor_ref())

        assignments = AssignmentCursor.from_function(self)
        for param, val in zip(self.func_params, inputs):
            assignments[param.name] = val
        vars, values = assignments.finalize()

        tensor_layer().wrap_kernel_call(self, vars, values)

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
        self_param = f"{self_modifier} {struct.to_slang_type()} self"
        param_definition = ", ".join([self_param] + params)
        param_use = ", ".join(p.name for p in self.method.params)

        prefix = "return " if not isinstance(return_type, VoidType) else ""
        code = (
            f"{return_type.to_slang_type()} {func_name}({param_definition}) {{\n"
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
