from .base import IOFlag, IVariable, Wrapper, register_wrapper, wrap, register_assignment, assign

from ..reflection import SlangType, IntIndexableType, StrIndexableType
from ..util import broadcast_shapes

from collections import OrderedDict
from collections.abc import Sequence

from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    from ..codegen import SlangCodeGen
    from ..layers import ShaderCursor


class RecursiveIndexable(IVariable):
    def __init__(self, result_type: SlangType, children: tuple[tuple[str, IVariable], ...], indices: tuple[int | str, ...]):
        super().__init__(result_type)
        self.children = children
        self.indices = indices

    def define(self, codegen: SlangCodeGen):
        for name, child in self.children:
            child.define(codegen)

        codegen.begin_block(f"struct {self} : Variable<{self.type}> {{")
        for name, child in self.children:
            codegen.emit(f"{child} {name};")

        fields = [f"[{i}]" if isinstance(i, int) else f".{i}" for i in self.indices]

        codegen.begin_block(f"{self.type} read(BatchIndex idx) {{")
        codegen.emit(f"{self.type} result;")
        for field, (name, _) in zip(fields, self.children):
            codegen.emit(f"result{field} = {name}.read(idx);")
        codegen.emit(f"return result;")
        codegen.end_block()

        codegen.begin_block(f"void write(BatchIndex idx, {self.type} value) {{")
        for field, (name, _) in zip(fields, self.children):
            codegen.emit(f"{name}.write(idx, value{field});")
        codegen.end_block()
        codegen.end_block(f"}}")

    def __hash__(self) -> int:
        return hash((RecursiveIndexable, self.type, self.indices, self.children))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RecursiveIndexable):
            return NotImplemented
        if self.type != other.type:
            return False

        return self.indices == other.indices and self.children == other.children


class DictWrapper(Wrapper):
    def __init__(self, result_type: StrIndexableType, value: dict[str, Any], flag: IOFlag):
        super().__init__()
        self.result_type = result_type

        fields = result_type.fields()
        indices = list(sorted(fields.keys()))

        for k in indices:
            if k not in fields:
                raise ValueError(f"Type {result_type} has no member {k}")

        self.children = []
        for name in indices:
            if name not in value:
                raise ValueError(
                    f"Argument is missing required struct member {type}::{k}")

            self.children.append((name, wrap(fields[name], value[name], flag)))

    def batch_size(self, value: dict[str, Any]) -> tuple[int, ...]:
        sizes = [c.batch_size(value[name]) for name, c in self.children]
        size = broadcast_shapes(sizes)
        assert size is not None  # TODO
        return size

    def broadcast(self, value: dict[str, Any], batch_size: tuple[int, ...]) -> tuple[IVariable, Any]:
        names = []
        vars = []
        out = OrderedDict()
        for name, child in self.children:
            var, val = child.broadcast(value[name], batch_size)
            vars.append((name, var))
            out[name] = val
            names.append(name)

        return RecursiveIndexable(self.result_type, tuple(vars), tuple(names)), out


class SequenceWrapper(Wrapper):
    def __init__(self, result_type: IntIndexableType, value: Sequence[Any], flag: IOFlag):
        super().__init__()
        self.result_type = result_type
        self.children = [wrap(result_type.element_type(), v, flag) for v in value]

        if len(self.children) != result_type.length():
            raise ValueError(f"Type {result_type} expects {result_type.length()} "
                             f"elements, received {len(self.children)} instead")

    def batch_size(self, value: Sequence[Any]) -> tuple[int, ...]:
        sizes = [c.batch_size(v) for c, v in zip(self.children, value)]
        size = broadcast_shapes(sizes)
        assert size is not None  # TODO
        return size

    def broadcast(self, value: Sequence[Any], batch_size: tuple[int, ...]) -> tuple[IVariable, Any]:
        recursed = [c.broadcast(v, batch_size) for c, v in zip(self.children, value)]

        indices = tuple(range(len(recursed)))
        fields = tuple((f"i{i}", var) for i, (var, _) in enumerate(recursed))
        out = [val for _, val in recursed]

        return RecursiveIndexable(self.result_type, fields, indices), out


def assign_indexable(type: RecursiveIndexable, var: ShaderCursor, value: Any):
    for i, (name, child) in zip(type.indices, type.children):
        assign(child, var[name], value[i])


register_wrapper(list, SequenceWrapper)
register_wrapper(tuple, SequenceWrapper)
register_wrapper(dict, DictWrapper)

register_assignment(RecursiveIndexable, assign_indexable)
