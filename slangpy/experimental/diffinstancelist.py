# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import numpy.typing as npt

from slangpy.core.instance import InstanceList
from slangpy.core.struct import Struct
from slangpy.experimental.diffbuffer import NDDifferentiableBuffer


class InstanceDifferentiableBuffer(InstanceList):
    """
    WIP: Differentiable buffer not currently supported for general types.

    Simplified implementation of InstanceList that uses a single differentiable buffer for all instances and
    provides buffer convenience functions for accessing its data.
    """

    def __init__(self, struct: Struct, shape: tuple[int, ...], data: Optional[NDDifferentiableBuffer] = None):
        if data is None:
            data = NDDifferentiableBuffer(struct.device_module.session.device,
                                          element_type=struct, shape=shape, requires_grad=True)
        super().__init__(struct, data)
        if data is None:
            data = {}

    @property
    def shape(self):
        """
        Get the shape of the buffer.
        """
        return self._data.shape

    @property
    def buffer(self):
        """
        Get the buffer.
        """
        return self._data

    def primal_to_numpy(self):
        """
        Convert the primal buffer to a numpy array.
        """
        return self.buffer.primal_to_numpy()

    def primal_from_numpy(self, data: npt.ArrayLike):
        """
        Set the primal buffer from a numpy array.
        """
        self.buffer.primal_from_numpy(data)

    def grad_to_numpy(self):
        """
        Convert the gradient buffer to a numpy array.
        """
        return self.buffer.grad_to_numpy()

    def grad_from_numpy(self, data: npt.ArrayLike):
        """
        Set the gradient buffer from a numpy array.
        """
        self.buffer.grad_from_numpy(data)
