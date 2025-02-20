# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from __future__ import annotations

from slangpy.backend import Device, DataType, TypeReflection
from slangpy.types import Tensor, NDBuffer
from typing import Optional, Any
from enum import Enum
import numpy as np
import math


# Root interface representing a slang type that implements the IModule interface
class NeuralModule:
    @property
    def input_type(self) -> str:
        raise NotImplementedError()

    @property
    def output_type(self) -> str:
        raise NotImplementedError()

    # Returns a dictionary containing the data for the slang struct
    def get_this(self) -> dict[str, Any]:
        raise NotImplementedError()

    # Returns a list of this module and all child modules
    def modules(self) -> list[NeuralModule]:
        return [self]

    def initialize(self, device: Device):
        pass

    def parameters(self) -> list[Tensor]:
        return []


# Chains multiple modules together into a new module
class ModuleChain(NeuralModule):
    def __init__(self, *modules: NeuralModule):
        super().__init__()

        if len(modules) == 0:
            raise ValueError("Module chain needs at least one module")

        self.chain = list(modules)

    @property
    def input_type(self) -> str:
        return self.chain[0].input_type

    @property
    def output_type(self) -> str:
        return self.chain[-1].output_type

    def initialize(self, device: Device):
        for m in self.chain:
            m.initialize(device)

    def parameters(self) -> list[Tensor]:
        return sum((m.parameters() for m in self.chain), start=[])

    def get_this(self):
        second_data = self.chain[-1].get_this()

        for first in reversed(self.chain[:-1]):
            first_data = first.get_this()
            type_name = ("ChainedModulePair<"
                         f"{first.input_type}, {first.output_type}, {self.output_type}, "
                         f"{first_data['_type']}, {second_data['_type']}>")

            second_data = {
                "_type": type_name,
                "first": first_data,
                "second": second_data
            }
        return second_data

    def modules(self) -> list[NeuralModule]:
        return sum((m.modules() for m in self.chain), start=[self])


class Real(Enum):
    half = 1
    float = 2
    double = 3

    def __str__(self):
        return self._name_

    def numpy(self):
        if self is Real.half:
            return np.float16
        elif self is Real.float:
            return np.float32
        elif self is Real.double:
            return np.float64
        else:
            raise ValueError(f"Invalid Real type '{self}'")

    def slang(self):
        if self is Real.half:
            return TypeReflection.ScalarType.float16
        elif self is Real.float:
            return TypeReflection.ScalarType.float32
        elif self is Real.double:
            return TypeReflection.ScalarType.float64
        else:
            raise ValueError(f"Invalid Real type '{self}'")

    def sgl(self):
        if self is Real.half:
            return DataType.float16
        elif self is Real.float:
            return DataType.float32
        elif self is Real.double:
            return DataType.float64
        else:
            raise ValueError(f"Invalid Real type '{self}'")

    def size(self):
        if self is Real.half:
            return 2
        elif self is Real.float:
            return 4
        elif self is Real.double:
            return 8
        else:
            raise ValueError(f"Invalid Real type '{self}'")


class ArrayModule(NeuralModule):
    def __init__(self, num_inputs: int, num_outputs: int, dtype: Real):
        super().__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.dtype = dtype

    @property
    def input_type(self) -> str:
        return f"{self.dtype}[{self.num_inputs}]"

    @property
    def output_type(self) -> str:
        return f"{self.dtype}[{self.num_outputs}]"


class FrequencyEncoding(ArrayModule):
    # Frequency encoding that maps each input parameter into a series
    # of sines and cosines with increasing frequency
    def __init__(self, input_width: int, num_octaves: int, dtype: Real = Real.float):
        if num_octaves == 0:
            raise ValueError("Missing argument: num_octaves")

        self.num_octaves = num_octaves
        output_width = num_octaves * input_width * 2

        super().__init__(input_width, output_width, dtype)

    def get_this(self):
        return {"_type": f"FrequencyEncoding<{self.dtype}, {self.num_inputs}, {self.num_octaves}>"}


# Root class for all activations (i.e. that implement IActivation)
class Activation(ArrayModule):
    def __init__(self, act_name: str, width: int, dtype: Real = Real.float):
        super().__init__(width, width, dtype)
        self.act_name = act_name

    def get_this(self) -> dict[str, Any]:
        return {"_type": f"Activation::{self.act_name}<{self.dtype}, {self.num_inputs}>"}


class NoneAct(Activation):
    def __init__(self, width: int, dtype: Real = Real.float):
        super().__init__("None", width, dtype)


class ReLUAct(Activation):
    def __init__(self, width: int, dtype: Real = Real.float):
        super().__init__("ReLU", width, dtype)


class LeakyReLUAct(Activation):
    def __init__(self, width: int, negative_slope: float = 0.01, dtype: Real = Real.float):
        super().__init__("LeakyReLU", width, dtype)
        self.negative_slope = negative_slope

    def get_this(self):
        return super().get_this() | {"negativeSlope": self.negative_slope}


class ELUAct(Activation):
    def __init__(self, width: int, a: float = 1.0, dtype: Real = Real.float):
        super().__init__("ELU", width, dtype)
        self.a = a

    def get_this(self):
        return super().get_this() | {"a": self.a}


class SwishAct(Activation):
    def __init__(self, width: int, dtype: Real = Real.float):
        super().__init__("Swish", width, dtype)


class TanhAct(Activation):
    def __init__(self, width: int, dtype: Real = Real.float):
        super().__init__("Tanh", width, dtype)


class SigmoidAct(Activation):
    def __init__(self, width: int, dtype: Real = Real.float):
        super().__init__("Sigmoid", width, dtype)


class ExpAct(Activation):
    def __init__(self, width: int, dtype: Real = Real.float):
        super().__init__("Exp", width, dtype)


class LinearLayer(ArrayModule):
    def __init__(self, num_inputs: int, num_outputs: int, dtype: Real = Real.float):
        super().__init__(num_inputs, num_outputs, dtype)

        self.weights: Optional[Tensor] = None
        self.biases: Optional[Tensor] = None

    def initialize(self, device: Device):
        # Xavier uniform initialization
        std = math.sqrt(2.0 / (self.num_inputs + self.num_outputs))
        a = math.sqrt(3.0) * std
        weights_np = np.random.uniform(-a, a, (self.num_outputs,
                                       self.num_inputs)).astype(self.dtype.numpy())
        biases_np = np.zeros((self.num_outputs, ), dtype=self.dtype.numpy())

        self.weights = Tensor.empty(device, weights_np.shape, str(self.dtype)).with_grads(zero=True)
        self.biases = Tensor.empty(device, biases_np.shape, str(self.dtype)).with_grads(zero=True)
        self.weights.storage.copy_from_numpy(weights_np)
        self.biases.storage.copy_from_numpy(biases_np)

    def parameters(self):
        return [self.weights, self.biases]

    def get_this(self):
        if self.weights is None:
            raise RuntimeError("LinearLayer is not initialized!")

        return {
            "weights": self.weights.storage,
            "biases": self.biases.storage,
            "weightGrads": self.weights.grad_out.storage,
            "biasGrads": self.biases.grad_out.storage,
            "_type": f"LinearLayer<{self.dtype}, {self.num_inputs}, {self.num_outputs}>"
        }
