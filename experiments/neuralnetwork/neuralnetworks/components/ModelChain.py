# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from ..basetypes import IModel, SlangType

from slangpy import Module

from typing import Optional


# Chains multiple modules together into a new module
class ModelChain(IModel):
    def __init__(self, *models: IModel):
        super().__init__()

        if len(models) == 0:
            self.model_error("Model chain needs at least one model")

        self.models = list(models)
        for m in self.models:
            m.set_parent(self)

        self.chain = []
        root = models[-1]
        for m in reversed(self.models[:-1]):
            root = ChainedModelPair(m, root)
            root.set_parent(self)
            self.chain.append(root)

        self.root = root

    def model_init(self, module: Module, input_type: SlangType):
        self.root.initialize(module, input_type)

    def resolve_input_type(self, module: Module):
        return self.root.resolve_input_type(module)

    def child_name(self, child: IModel) -> Optional[str]:
        for i, m in enumerate(self.models):
            if m is child:
                return f"models[{i}]"
        for i, m in enumerate(self.chain):
            if m is child:
                return f"chain[{i}]"
        return None

    @property
    def type_name(self) -> str:
        return self.root.type_name

    def get_this(self):
        return self.root.get_this()

    def children(self) -> list[IModel]:
        return [self.root]


class ChainedModelPair(IModel):
    def __init__(self, first: IModel, second: IModel):
        super().__init__()

        self.first = first
        self.second = second

    def model_init(self, module: Module, input_type: SlangType):
        self.first.initialize(module, input_type)
        self.second.initialize(module, self.first.output_type)

    def resolve_input_type(self, module: Module):
        return self.first.resolve_input_type(module)

    @property
    def type_name(self) -> str:
        return ("ChainedModelPair<"
                f"{self.first.input_type.full_name}, "
                f"{self.first.output_type.full_name}, "
                f"{self.second.output_type.full_name}, "
                f"{self.first.type_name}, {self.second.type_name}>")

    def get_this(self):
        return {
            "_type": self.type_name,
            "first": self.first.get_this(),
            "second": self.second.get_this()
        }

    def children(self):
        return [self.first, self.second]
