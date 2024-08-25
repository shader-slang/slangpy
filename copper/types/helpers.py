from __future__ import annotations

from .interfaces import SlangType, TypeTranslator, CodegenType
from .indexables import StrIndexableType

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..variable import Variable
    from ..codegen import SlangCodeGen


class CollectionTranslator(TypeTranslator):
    def __init__(self, initializer: Optional[str] = None):
        super().__init__()
        self.initializer = "" if initializer is None else " = " + initializer

    def declare_variable(self, var: Variable, gen: SlangCodeGen):
        index = lambda idx: f".{idx}" if isinstance(idx, str) else f"[{idx}]"

        gen.begin_block("struct ")
        for subvar in var.children.values():
            subvar.declare(gen)
        gen.begin_block(f"{var.type.to_slang_type()} get(BatchIndex batchIdx) ")
        gen.emit(f"{var.type.to_slang_type()} result{self.initializer};")
        for idx, subvar in var.children.items():
            if subvar.is_in:
                gen.emit(f"result{index(idx)} = {subvar.read()};")
        gen.emit("return result;")
        gen.end_block()
        if var.is_out:
            gen.begin_block(
                f"void set(BatchIndex batchIdx, {var.type.to_slang_type()} value) "
            )
            for idx, subvar in var.children.items():
                if subvar.is_out:
                    subvar.write(subvar.name, f"value{index(idx)}", gen)
            gen.end_block()
        gen.end_block(f" {var.name};")

    def read_variable(self, var: Variable, var_name: str) -> str:
        return var_name + ".get(batchIdx)"

    def write_variable(
        self, var: Variable, var_name: str, value: str, gen: SlangCodeGen
    ):
        gen.emit(f"{var_name}.set(batchIdx, {value});")


class InlineStructType(CodegenType, StrIndexableType):
    def __init__(self, name: str, members: dict[str, SlangType]):
        super().__init__()
        self.name = name
        self.members = members

    def fields(self):
        return self.members

    def to_slang_type(self) -> str:
        return self.name

    def define_type(self, gen: SlangCodeGen):
        gen.begin_block(f"struct {self.name} ")
        for name, type in self.members.items():
            assert not isinstance(type, TypeTranslator)
            gen.emit(f"{type.to_slang_type()} {name};")
        gen.end_block()
