# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from ..basetypes import IModel, Real, ArrayKind, RealArray, TypeLike, Auto, AutoSettable

from sgl import CoopVecMatrixLayout
from slangpy import Module, Tensor

from typing import Optional, cast
import numpy as np
import math


class LinearLayer(IModel):
    def __init__(self, num_inputs: AutoSettable[int], num_outputs: int, dtype: AutoSettable[Real] = Auto, kind: AutoSettable[ArrayKind] = Auto):
        super().__init__()

        self.input_array = RealArray(kind, dtype, num_inputs)
        self.output_array = RealArray(kind, dtype, num_outputs)

        self.weights: Optional[Tensor] = None
        self.biases: Optional[Tensor] = None

    def initialize(self, module: Module, input_type: Optional[TypeLike]):
        input_array = RealArray.from_slangtype(input_type)
        self.input_array.resolve(input_array, must_match=True)
        self.output_array.resolve(self.input_array)

        if self.input_array.kind not in (ArrayKind.array, ArrayKind.coopvec):
            self.model_error("LinearLayer only supports arrays or CoopVec"
                             f" as input; received {self.input_array}")
        if self.input_array.dtype != self.output_array.dtype or self.input_array.kind != self.output_array.kind:
            self.model_error("Input and output type of LinearLayer must match; "
                             f"have '{self.input_array}' and '{self.output_array}'")

        if self.input_array.kind == ArrayKind.coopvec:
            if "cooperative-vector" not in module.device.features:
                self.model_error("Device does not support CoopVec")
            self.use_coopvec = True
        else:
            self.use_coopvec = False

        self.input_type = self.lookup_mandatory_type(module, self.input_array.name())
        self.output_type = self.lookup_mandatory_type(module, self.output_array.name())

        fan_in = self.input_array.length
        fan_out = self.output_array.length
        dtype = self.input_array.dtype

        # Xavier uniform initialization
        std = math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        weights_np = np.random.uniform(-a, a, (fan_out, fan_in)).astype(dtype.numpy())
        biases_np = np.zeros((fan_out, ), dtype=dtype.numpy())

        device = module.device
        biases = Tensor.empty(device, biases_np.shape, str(dtype))
        biases.storage.copy_from_numpy(biases_np)

        if self.use_coopvec:
            layout = CoopVecMatrixLayout.training_optimal
            desc = device.coopvec_create_matrix_desc(fan_out, fan_in, layout, dtype.sgl(), 0)
            weight_count = desc.size // dtype.size()

            params_np = np.zeros((weight_count, ), dtype=dtype.numpy())
            device.coopvec_convert_matrix_host(weights_np, params_np, dst_layout=layout)

            weights = Tensor.empty(device, (weight_count, ), str(dtype))
            weights.storage.copy_from_numpy(params_np)
        else:
            weights = Tensor.empty(device, weights_np.shape, str(dtype))
            weights.storage.copy_from_numpy(weights_np)

        weights = weights.with_grads(zero=True)
        biases = biases.with_grads(zero=True)

        self.weights = cast(Tensor, weights)
        self.biases = cast(Tensor, biases)

        self.validate(module)

    def parameters(self):
        if self.weights is None or self.biases is None:
            raise RuntimeError("LinearLayer is not initialized!")

        return [self.weights, self.biases]

    @property
    def type_name(self) -> str:
        base_type = "CoopVecLinearLayer" if self.use_coopvec else "LinearLayer"
        return f"{base_type}<{self.input_array.dtype}, {self.input_array.length}, {self.output_array.length}>"

    def get_this(self):
        if self.weights is None or self.biases is None:
            raise RuntimeError("LinearLayer is not initialized!")

        weight_grads = None
        if self.weights.grad_out:
            weight_grads = self.weights.grad_out.storage
        bias_grads = None
        if self.biases.grad_out:
            bias_grads = self.biases.grad_out.storage

        return {
            "weights": self.weights.storage,
            "biases": self.biases.storage,
            "weightGrads": weight_grads,
            "biasGrads": bias_grads,
            "_type": self.type_name
        }
