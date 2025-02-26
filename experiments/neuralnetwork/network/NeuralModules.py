# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from __future__ import annotations

from slangpy.backend import Device, DataType
from slangpy.types import Tensor, NDBuffer
from typing import Optional, Any
import numpy as np
import math


# Root interface representing a slang type that implements the IModule interface
class NeuralModule:
    def __init__(self, fan_in: int, fan_out: int, dtype: DataType = DataType.float32):
        super().__init__()

        self.dtype = dtype
        self.fan_in = fan_in
        self.fan_out = fan_out

    # Returns a dictionary containing the data for the slang struct
    def get_this(self) -> dict[str, Any]:
        raise NotImplementedError()

    @property
    def type_name(self):
        return self.get_this()["_type"]

    # Returns a list of this module and all child modules
    def modules(self) -> list[NeuralModule]:
        return [self]

    def initialize(self, device):
        pass

    def parameters(self) -> list[Tensor]:
        return []


# Chains multiple modules together into a new module
class ModuleChain(NeuralModule):
    def __init__(self, *modules: NeuralModule):
        if len(modules) == 0:
            raise ValueError("Module chain needs at least one module")

        self.chain = list(modules)

        super().__init__(modules[0].fan_in, modules[-1].fan_out, modules[0].dtype)

    def initialize(self, device):
        for m in self.chain:
            m.initialize(device)

    def parameters(self) -> list[Tensor]:
        return sum((m.parameters() for m in self.chain), start=[])

    def get_this(self):
        result = self.chain[-1].get_this()

        for m in reversed(self.chain[:-1]):
            first = m.get_this()
            type_name = ("ModuleChain<"
                         f"{dtype_name(self.dtype)}, "
                         f"{m.fan_in}, {m.fan_out}, {self.fan_out}, "
                         f"{first['_type']}, {result['_type']}> ")
            result = {
                "_type": type_name,
                "first": first,
                "second": result
            }
        return result

    def modules(self) -> list[NeuralModule]:
        return sum((m.modules() for m in self.chain), start=[self])


# Frequency encoding that maps each input parameter into a series
# of sines and cosines with increasing frequency
class FrequencyEncoding(NeuralModule):
    def __init__(self, input_width: int, num_octaves: int, dtype: DataType = DataType.float32):
        if num_octaves == 0:
            raise ValueError("Missing argument: num_octaves")

        self.num_octaves = num_octaves
        output_width = num_octaves * input_width * 2

        super().__init__(input_width, output_width, dtype)

    def get_this(self):
        return {"_type": f"FrequencyEncoding<{dtype_name(self.dtype)}, {self.fan_in}, {self.num_octaves}> "}


# Root class for all activations (i.e. that implement IActivation)
class Activation(NeuralModule):
    def __init__(self, act_name: str, width: int, dtype: DataType = DataType.float32):
        super().__init__(width, width, dtype)
        self.act_name = act_name

    def get_this(self) -> dict[str, Any]:
        return {"_type": f"Activation::{self.act_name}<{dtype_name(self.dtype)}, {self.fan_in}> "}


class NoneAct(Activation):
    def __init__(self, width: int, dtype: DataType = DataType.float32):
        super().__init__("None", width, dtype)


class ReLUAct(Activation):
    def __init__(self, width: int, dtype: DataType = DataType.float32):
        super().__init__("ReLU", width, dtype)


class LeakyReLUAct(Activation):
    def __init__(self, width: int, negative_slope: float = 0.01, dtype: DataType = DataType.float32):
        super().__init__("LeakyReLU", width, dtype)
        self.negative_slope = negative_slope

    def get_this(self):
        return super().get_this() | {"negativeSlope": self.negative_slope}


class ELUAct(Activation):
    def __init__(self, width: int, a: float = 1.0, dtype: DataType = DataType.float32):
        super().__init__("ELU", width, dtype)
        self.a = a

    def get_this(self):
        return super().get_this() | {"a": self.a}


class SwishAct(Activation):
    def __init__(self, width: int, dtype: DataType = DataType.float32):
        super().__init__("Swish", width, dtype)


class TanhAct(Activation):
    def __init__(self, width: int, dtype: DataType = DataType.float32):
        super().__init__("Tanh", width, dtype)


class SigmoidAct(Activation):
    def __init__(self, width: int, dtype: DataType = DataType.float32):
        super().__init__("Sigmoid", width, dtype)


class ExpAct(Activation):
    def __init__(self, width: int, dtype: DataType = DataType.float32):
        super().__init__("Exp", width, dtype)


class LinearLayer(NeuralModule):
    def __init__(self, fan_in: int, fan_out: int, dtype: DataType = DataType.float32):
        super().__init__(fan_in, fan_out, dtype)

        self.weights: Optional[Tensor] = None
        self.biases: Optional[Tensor] = None

    def initialize(self, device: Device):
        # Xavier uniform initialization
        std = math.sqrt(2.0 / (self.fan_in + self.fan_out))
        a = math.sqrt(3.0) * std
        weights_np = np.random.uniform(-a, a, (self.fan_out, self.fan_in)
                                       ).astype(dtype_to_numpy(self.dtype))
        biases_np = np.zeros((self.fan_out, ), dtype=dtype_to_numpy(self.dtype))

        self.weights = Tensor.empty(device, weights_np.shape,
                                    dtype_name(self.dtype)).with_grads(zero=True)
        self.biases = Tensor.empty(device, biases_np.shape,
                                   dtype_name(self.dtype)).with_grads(zero=True)
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
            "_type": f"LinearLayer<{dtype_name(self.dtype)}, {self.fan_in}, {self.fan_out}> "
        }


def dtype_name(dtype: DataType):
    if dtype == DataType.float16:
        return "half"
    elif dtype == DataType.float32:
        return "float"
    else:
        raise ValueError(f"Unsupported CoopVec datatype '{dtype}'")


def dtype_to_numpy(dtype: DataType):
    if dtype == DataType.float16:
        return np.float32
    elif dtype == DataType.float32:
        return np.float32
    else:
        raise ValueError(f"Unsupported CoopVec datatype '{dtype}'")
