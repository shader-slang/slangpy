# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from __future__ import annotations

from slangpy.backend import Device, CoopVecMatrixLayout
from slangpy.types import Tensor
from typing import Optional, Union
import numpy as np
import math

from .NeuralModules import NeuralModule, ArrayModule, Real, AutoSettable, Auto


class CoopVecModule(ArrayModule):
    def __init__(self, num_inputs: AutoSettable[int], num_outputs: AutoSettable[int], dtype: AutoSettable[Real]):
        super().__init__(num_inputs, num_outputs, dtype)

    @property
    def input_type(self) -> str:
        return f"DiffCoopVec<{self.dtype}, {self.num_inputs}>"

    @property
    def output_type(self) -> str:
        return f"DiffCoopVec<{self.dtype}, {self.num_outputs}>"


class CoopVecToArray(CoopVecModule):
    def __init__(self, num_inputs: AutoSettable[int] = Auto, dtype: AutoSettable[Real] = Auto):
        super().__init__(num_inputs, num_inputs, dtype)

    @property
    def output_type(self) -> str:
        return f"{self.dtype}[{self.num_outputs}]"

    def set_inputs(self, inputs: Union[None, list[NeuralModule]]):
        super().set_inputs(inputs)
        self.num_outputs = self.num_inputs

    def get_this(self):
        return {"_type": f"CoopVecToArray<{self.dtype}, {self.num_inputs}>"}


class ArrayToCoopVec(CoopVecModule):
    def __init__(self, num_inputs: AutoSettable[int] = Auto, dtype: AutoSettable[Real] = Auto):
        super().__init__(num_inputs, num_inputs, dtype)

    @property
    def input_type(self) -> str:
        return f"{self.dtype}[{self.num_outputs}]"

    def set_inputs(self, inputs: Union[None, list[NeuralModule]]):
        super().set_inputs(inputs)
        self.num_outputs = self.num_inputs

    def get_this(self):
        return {"_type": f"ArrayToCoopVec<{self.dtype}, {self.num_inputs}>"}


class LinearLayer(CoopVecModule):
    def __init__(self, num_inputs: AutoSettable[int], num_outputs: int, dtype: AutoSettable[Real] = Auto):
        super().__init__(num_inputs, num_outputs, dtype)

        self.params: Optional[Tensor] = None

    def initialize(self, device: Device):
        if "cooperative-vector" not in device.features:
            raise RuntimeError("Device does not support CoopVec")

        fan_in = self.num_inputs
        fan_out = self.num_outputs
        cur_offset = 0
        elem_size = self.dtype.size()

        cur_offset = device.coopvec_align_matrix_offset(cur_offset)
        weight_offset = cur_offset // elem_size
        desc = device.coopvec_create_matrix_desc(
            fan_out, fan_in, CoopVecMatrixLayout.training_optimal, self.dtype.sgl(), cur_offset)
        cur_offset += desc.size
        weight_count = desc.size // elem_size

        cur_offset = device.coopvec_align_vector_offset(cur_offset)
        bias_offset = cur_offset // elem_size
        cur_offset += fan_out * elem_size
        bias_count = fan_out

        param_count = cur_offset // elem_size
        self.params = Tensor.empty(device, (param_count, ), str(self.dtype)).with_grads(zero=True)
        self.offsets = [weight_offset, bias_offset]

        # Xavier uniform initialization
        std = math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        weights_np = np.random.uniform(-a, a, (fan_out, fan_in)).astype(self.dtype.numpy())
        biases_np = np.zeros((fan_out, ), dtype=self.dtype.numpy())

        params_np = np.zeros((param_count, ), dtype=self.dtype.numpy())
        coopvec_weights_np = params_np[weight_offset:weight_offset + weight_count]
        coopvec_biases_np = params_np[bias_offset:bias_offset + bias_count]

        device.coopvec_convert_matrix_host(
            weights_np, coopvec_weights_np, dst_layout=CoopVecMatrixLayout.training_optimal)
        coopvec_biases_np[:] = biases_np[:]

        self.params.storage.copy_from_numpy(params_np)

    def get_this(self):
        if self.params is None:
            raise RuntimeError("LinearLayer is not initialized!")

        return {
            "parameters": self.params.storage,
            "gradients": self.params.grad_out.storage,
            "offsets": self.offsets,
            "_type": f"CoopVecLinearLayer<{self.dtype}, {self.num_inputs}, {self.num_outputs}> "
        }

    def parameters(self) -> list[Tensor]:
        return [self.params]


# Root class for all coopvec activations (i.e. that implement IActivation)
class Activation(CoopVecModule):
    def __init__(self, act_name: str, width: int, dtype: AutoSettable[Real] = Auto):
        super().__init__(width, width, dtype)
        self.act_name = act_name

    def set_inputs(self, inputs: Union[None, list[NeuralModule]]):
        super().set_inputs(inputs)
        self.num_outputs = self.num_inputs

    def get_this(self):
        return {"_type": f"CoopVecAct::{self.act_name}<{self.dtype}, {self.num_inputs}>"}


class NoneAct(Activation):
    def __init__(self, width: int, dtype: AutoSettable[Real] = Auto):
        super().__init__("None", width, dtype)


class ReLUAct(Activation):
    def __init__(self, width: int, dtype: AutoSettable[Real] = Auto):
        super().__init__("ReLU", width, dtype)


class LeakyReLUAct(Activation):
    def __init__(self, width: int, negative_slope: float = 0.01, dtype: AutoSettable[Real] = Auto):
        super().__init__("LeakyReLU", width, dtype)
        self.negative_slope = negative_slope

    def get_this(self):
        return super().get_this() | {"negativeSlope": self.negative_slope}


class ELUAct(Activation):
    def __init__(self, width: int, a: float = 1.0, dtype: AutoSettable[Real] = Auto):
        super().__init__("ELU", width, dtype)
        self.a = a

    def get_this(self):
        return super().get_this() | {"a": self.a}


class SwishAct(Activation):
    def __init__(self, width: int, dtype: AutoSettable[Real] = Auto):
        super().__init__("Swish", width, dtype)


class TanhAct(Activation):
    def __init__(self, width: int, dtype: AutoSettable[Real] = Auto):
        super().__init__("Tanh", width, dtype)


class SigmoidAct(Activation):
    def __init__(self, width: int, dtype: AutoSettable[Real] = Auto):
        super().__init__("Sigmoid", width, dtype)


class ExpAct(Activation):
    def __init__(self, width: int, dtype: AutoSettable[Real] = Auto):
        super().__init__("Exp", width, dtype)
