from __future__ import annotations

import logging
from typing import Optional, Any

from collections import OrderedDict

from .codegen import SlangCodeGen
from .types.interfaces import SlangType, TypeTranslator, TypedValue, DontCare
from .types.indexables import IntIndexableType, StrIndexableType
from .types.helpers import InlineStructType, CollectionTranslator
from .types.tensor import TensorType, TensorKind
from .layers import tensor_layer, TensorRef

from typing import NoReturn, TYPE_CHECKING

if TYPE_CHECKING:
    from .invokables import InvokableSlangFunc


class VariableError(Exception):
    def __init__(self, path: str, msg: str):
        self.path = path
        self.msg = msg
        super().__init__()

    def __str__(self):
        return f"Error while processing parameter '{self.path}': {self.msg}"


class Variable:
    def __init__(
        self,
        name: str,
        type: SlangType,
        is_in: bool,
        is_out: bool,
        is_differentiable: bool,
    ):
        super().__init__()

        assert is_in or is_out

        self.parent: Optional[Variable] = None
        self.name = name
        self.type = type
        self.translator: Optional[TypeTranslator] = None
        self.is_in = is_in
        self.is_out = is_out
        self.is_differentiable = is_differentiable
        self.is_leaf = True
        self.children: dict[int | str, Variable] = OrderedDict()

    def __getitem__(self, idx: int | str) -> Variable:
        return self.children[idx]

    def __contains__(self, key: int | str) -> bool:
        return key in self.children

    def leaves(self):
        if self.is_leaf:
            yield self
        else:
            for v in self.children.values():
                yield from v.leaves()

    def nodes(self):
        yield self

        for v in self.children.values():
            yield from v.nodes()

    def declare(self, gen: SlangCodeGen):
        if self.translator is not None:
            self.translator.declare_variable(self, gen)
        else:
            gen.emit(f"{self.type.to_slang_type()} {self.name};")

    def read(self, var_name: Optional[str] = None) -> str:
        if var_name is None:
            var_name = self.name

        if self.translator is not None:
            return self.translator.read_variable(self, var_name)
        else:
            return var_name

    def write(self, var_name: Optional[str], value: str, gen: SlangCodeGen):
        if var_name is None:
            var_name = self.name

        if self.translator is not None:
            self.translator.write_variable(self, var_name, value, gen)
        else:
            gen.emit(f"{var_name} = {value};")

    def path(self) -> tuple[str | int, ...]:
        if isinstance(self.parent, Variable):
            for k, v in self.parent.children.items():
                if v is self:
                    return self.parent.path() + (k,)

            raise RuntimeError(
                f"Bad internal state: Variable {self.name} is not listed in its parent."
            )
        else:
            return tuple()

    def get_root_path_string(self) -> str:
        path = self.path()
        if path:
            result = str(path[0])
            for p in path[1:]:
                if isinstance(p, str):
                    result += "." + p
                else:
                    result += "[" + str(p) + "]"
            return result
        else:
            return self.name

    def error(self, msg: str) -> NoReturn:
        raise VariableError(self.get_root_path_string(), msg)

    def __repr__(self):
        def format_var(var: Variable, indent: str = ""):
            flags = ""
            flags += "i" if var.is_in else " "
            flags += "o" if var.is_out else " "
            flags += "d" if var.is_differentiable else " "
            flags += "l" if var.is_leaf else " "

            result = [[indent, var.type.to_slang_type(), var.name, f"[{flags}]"]]

            for subvar in var.children.values():
                result += format_var(subvar, indent + "  ")

            return result

        lines = format_var(self)

        max_len = {}

        for line in lines:
            for i in range(1, len(line)):
                key = (line[0], i)
                max_len[key] = max(max_len.get(key, 0), len(line[i]))

        for line in lines:
            for i in range(1, len(line)):
                line[i] += " " * (max_len[(line[0], i)] - len(line[i]))

        return "\n".join((" ".join(line) for line in lines))


class AssignmentCursor(Variable):
    def __init__(
        self,
        parent: AssignmentCursor | InvokableSlangFunc,
        name: str,
        type: SlangType,
        is_in: bool,
        is_out: bool,
        is_differentiable: bool,
    ):
        super().__init__(name, type, is_in, is_out, is_differentiable)
        self.parent: AssignmentCursor | InvokableSlangFunc = parent
        self.children: dict[int | str, AssignmentCursor] = OrderedDict()
        self.value = None

    def set(self, value: Any):
        if isinstance(value, DontCare):
            if self.is_in:
                self.error("DontCare is only allowed for out parameters")
            self.prune()
            return

        type = self.type
        if isinstance(value, TypedValue):
            type = value.get_type()
        elif tensor_layer().is_tensor(value):
            value = tensor_layer().wrap_tensor(value)
            ndim = len(value.get_shape())
            type = TensorType(TensorKind.Tensor, value.get_dtype(), ndim)

        self.value = value
        type.assign_vars(self, value)

        if self.is_leaf and self.is_out and not isinstance(self.value, TensorRef):
            self.error("All parameters marked as 'out' must be assigned a tensor")

    def set_translator(self, translator: TypeTranslator):
        self.translator = translator

    def create_sub_assignment(
        self, idx: int | str, is_in: bool, is_out: bool, is_differentiable: bool
    ):
        if isinstance(idx, int):
            if not isinstance(self.type, IntIndexableType):
                raise ValueError(
                    f"Slang type {self.type.to_slang_type()} is not indexable"
                )
            if idx < 0 or idx >= self.type.length():
                raise ValueError(
                    f"Index {idx} is out of range for slang type {self.type.to_slang_type()}"
                )

            subtype = self.type.element_type()
        elif isinstance(idx, str):
            if not isinstance(self.type, StrIndexableType):
                raise ValueError(
                    f"Slang type {self.type.to_slang_type()} has no named members"
                )
            if idx not in self.type.fields():
                raise ValueError(
                    f'Slang type {self.type.to_slang_type()} has no member "{idx}"'
                )

            subtype = self.type.fields()[idx]

        is_differentiable = is_differentiable and subtype.differentiable()
        name = idx if isinstance(idx, str) else "idx_" + str(idx)

        self.children[idx] = AssignmentCursor(
            self, name, subtype, is_in, is_out, is_differentiable
        )
        self.is_leaf = False

    def __getitem__(self, idx: int | str) -> AssignmentCursor:
        if idx not in self.children:
            self.create_sub_assignment(
                idx, self.is_in, self.is_out, self.is_differentiable
            )

        return self.children[idx]

    def __setitem__(self, idx: int | str, val: Any):
        self[idx].set(val)

    def prune(self):
        if isinstance(self.parent, AssignmentCursor):
            for k, v in self.parent.children.items():
                if v is self:
                    del self.parent.children[k]
                    return

            raise RuntimeError(
                f"Bad internal state: AssignmentCursor is not listed in its parent."
            )
        else:
            raise RuntimeError(
                f"Trying to prune root-level assignment (of function {self.parent.func_name})"
            )

    def finalize(self) -> tuple[Variable, list[Any]]:
        values = [self.value]
        var = Variable(
            self.name, self.type, self.is_in, self.is_out, self.is_differentiable
        )

        for idx, child in self.children.items():
            subvar, subvals = child.finalize()
            subvar.parent = var
            var.children[idx] = subvar
            values.extend(subvals)

        needs_translation = any(
            subvar.translator is not None for subvar in var.children.values()
        )
        if needs_translation and var.translator is None:
            var.translator = CollectionTranslator()
            logging.debug(
                f"Translating {self.get_root_path_string()} ({self.type.to_slang_type()})"
            )

        return var, values

    @staticmethod
    def from_function(func: InvokableSlangFunc) -> AssignmentCursor:
        type = InlineStructType(
            "InstanceArguments",
            OrderedDict([(param.name, param.type) for param in func.func_params]),
        )

        root = AssignmentCursor(func, "rootArgs", type, True, True, True)
        for p in func.func_params:
            root.create_sub_assignment(p.name, p.is_in(), p.is_out(), not p.is_nodiff())

        return root
