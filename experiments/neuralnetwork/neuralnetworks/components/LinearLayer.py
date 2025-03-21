# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from ..basetypes import IModel, Real, ArrayKind, RealArray, SlangType, Auto, AutoSettable, resolve_auto

from sgl import CoopVecMatrixLayout
from slangpy import Module, Tensor

from typing import cast, Any
import numpy as np
import math


class LinearLayer(IModel):
    """
    Represents a linear neural network layer, i.e. a matrix-vector multiply/add A * x + b

    Takes an input with num_inputs elements, and returns num_outputs elements.
    Currently implemented for plain arrays and CoopVec, the latter using hardware-assisted
    matrix multiplies.

    Due to limitations with atomics, dtype must be float if array inputs are used.
    Due to limitations with current CoopVec implementations, dtype must be half if
    CoopVec inputs are used.

    Don't use model_params directly, as the parameter tensors may be in a hardware-specific
    layout if CoopVec is used. Instad, use get_weights or get_biases to get the weights
    and biases as a row-major numpy array.
    """

    def __init__(self, num_inputs: AutoSettable[int], num_outputs: int, dtype: AutoSettable[Real] = Auto, use_coopvec: AutoSettable[bool] = Auto):
        super().__init__()

        self.num_outputs = num_outputs
        self._num_inputs = num_inputs
        self._dtype = dtype
        self._use_coopvec = use_coopvec

    def get_weights(self) -> np.ndarray[Any, Any]:
        self.check_initialized()
        weights_np = self.weights.to_numpy()

        if self.use_coopvec:
            rowmaj_weights = np.empty((self.num_outputs, self.num_inputs), dtype=self.dtype.numpy())

            layout = CoopVecMatrixLayout.training_optimal
            self.weights.device.coopvec_convert_matrix_host(
                weights_np, rowmaj_weights, src_layout=layout)

            return rowmaj_weights
        else:
            return weights_np

    def get_biases(self) -> np.ndarray[Any, Any]:
        self.check_initialized()
        return self.biases.to_numpy()

    @property
    def type_name(self) -> str:
        base_type = "CoopVecLinearLayer" if self.use_coopvec else "LinearLayer"
        return f"{base_type}<{self.dtype}, {self.num_inputs}, {self.num_outputs}>"

    def model_init(self, module: Module, input_type: SlangType):
        input_array = RealArray.from_slangtype(input_type)
        self.num_inputs = resolve_auto(self._num_inputs, input_array.length)
        self.dtype = resolve_auto(self._dtype, input_array.dtype)
        self.use_coopvec = resolve_auto(self._use_coopvec, input_array.kind == ArrayKind.coopvec)

        if input_array.kind not in (ArrayKind.array, ArrayKind.coopvec):
            self.model_error("LinearLayer only supports arrays or CoopVec as input type. "
                             f"Received {input_array}")

        if self.use_coopvec:
            if self.dtype != Real.half:
                self.model_error("LinearLayer currently only supports half precision as input "
                                 f"when using CoopVec. Received {input_array}")

            if "cooperative-vector" not in module.device.features:
                self.model_error("LinearLayer was requested to use the CoopVec API, "
                                 "but the device does not support it.")
        else:
            if self.dtype not in (Real.float, Real.double):
                self.model_error("LinearLayer currently only supports float or double precision "
                                 f"as input when not using CoopVec. Received {input_array}")

        # Xavier uniform initialization
        fan_in = self.num_inputs
        fan_out = self.num_outputs
        std = math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        weights_np = np.random.uniform(-a, a, (fan_out, fan_in)).astype(self.dtype.numpy())
        biases_np = np.zeros((fan_out, ), dtype=self.dtype.numpy())

        device = module.device
        biases = Tensor.empty(device, biases_np.shape, str(self.dtype))
        biases.storage.copy_from_numpy(biases_np)

        if self.use_coopvec:
            layout = CoopVecMatrixLayout.training_optimal
            desc = device.coopvec_create_matrix_desc(fan_out, fan_in, layout, self.dtype.sgl(), 0)
            weight_count = desc.size // self.dtype.size()

            params_np = np.zeros((weight_count, ), dtype=self.dtype.numpy())
            device.coopvec_convert_matrix_host(weights_np, params_np, dst_layout=layout)

            weights = Tensor.empty(device, (weight_count, ), str(self.dtype))
            weights.storage.copy_from_numpy(params_np)
        else:
            weights = Tensor.empty(device, weights_np.shape, str(self.dtype))
            weights.storage.copy_from_numpy(weights_np)

        weights = weights.with_grads(zero=True)
        biases = biases.with_grads(zero=True)

        self.weights = cast(Tensor, weights)
        self.biases = cast(Tensor, biases)

    def model_params(self):
        return [self.weights, self.biases]

    def resolve_input_type(self, module: Module):
        if self._num_inputs is Auto:
            return None

        return RealArray(
            ArrayKind.array,
            resolve_auto(self._dtype, Real.float),
            self._num_inputs
        )

    def get_this(self):
        self.check_initialized()

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
