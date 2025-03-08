# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from ..basetypes import IModel, TypeLike

from slangpy import Module, Tensor

from typing import Optional


# Chains multiple modules together into a new module
class ModelChain(IModel):
    def __init__(self, *models: IModel):
        super().__init__()

        if len(models) == 0:
            raise ValueError("Model chain needs at least one model")

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

    def child_name(self, child: IModel) -> Optional[str]:
        for i, m in enumerate(self.models):
            if m is child:
                return f"models[{i}]"
        for i, m in enumerate(self.chain):
            if m is child:
                return f"chain[{i}]"
        return None

    def initialize(self, module: Module, input_type: Optional[TypeLike]):
        self.root.initialize(module, input_type)

        self.input_type = self.root.input_type
        self.output_type = self.root.output_type
        self.validate(module)

    def parameters(self) -> list[Tensor]:
        return self.root.parameters()

    @property
    def type_name(self) -> str:
        return self.root.type_name

    def get_this(self):
        return self.root.get_this()

    def modules(self) -> list[IModel]:
        return self.root.modules()


class ChainedModelPair(IModel):
    def __init__(self, first: IModel, second: IModel):
        super().__init__()

        self.first = first
        self.second = second

    def initialize(self, module: Module, input_type: Optional[TypeLike]):
        self.first.initialize(module, input_type)
        self.second.initialize(module, self.first.output_type)

        self.input_type = self.first.input_type
        self.output_type = self.second.output_type
        self.validate(module)

    def parameters(self) -> list[Tensor]:
        return self.first.parameters() + self.second.parameters()

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

    def modules(self) -> list[IModel]:
        return [self] + self.first.modules() + self.second.modules()
