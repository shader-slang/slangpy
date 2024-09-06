from typing import Any, Optional
from numpy import ndarray
from sgl import Buffer, Device
from kernelfunctions.typeregistry import get_python_type_marshall, register_python_type
from kernelfunctions.types import PythonMarshal
from kernelfunctions.types.enums import AccessType


class ScalarRef:
    """
    Minimal class to hold a reference to a scalar value, allowing user to get outputs
    from scalar inout/out arguments.
    """

    def __init__(self, init_value: Any):
        super().__init__()
        self.value = init_value


class ScalarRefMarshall(PythonMarshal):
    """
    Marshall for scalar ref. Mainly works by passing through calls to the
    value's marshall.
    """

    def __init__(self):
        super().__init__(ScalarRef)

    def get_element_shape(self, value: ScalarRef):
        return get_python_type_marshall(value.value).get_element_shape(value.value)

    def get_element_type(self, value: ScalarRef):
        return get_python_type_marshall(value.value).get_element_type(value.value)

    def is_writable(self, value: ScalarRef) -> bool:
        return True

    def primal_to_numpy(self, value: ScalarRef):
        """
        For writing to a buffer, can just override the to_numpy function to get 
        the marshall for the value call its primal_to_numpy
        """
        return get_python_type_marshall(value.value).primal_to_numpy(value.value)

    def create_primal_calldata(self, device: Device, value: ScalarRef, access: AccessType):
        """
        Return entry in call data for the primal value. For write, can rely on default
        buffer behaviour, but for reading need to return inner value
        """
        if access == AccessType.read:
            return value.value
        else:
            return super().create_primal_calldata(device, value, access)

    def read_primal_calldata(self, device: Device, call_data: Any, access: AccessType, value: ScalarRef):
        """
        For readback, need to use the value's marshal to convert numpy array, then
        store the result in the value field.
        """
        if access != AccessType.read:
            assert isinstance(call_data, Buffer)
            numpy_value = call_data.to_numpy()
            primal = get_python_type_marshall(value.value).primal_from_numpy(numpy_value)
            value.value = primal

    def allocate_return_value(self, device: Device, call_shape: list[int], element_type: Any):
        """
        Allocate a return value for this type. Only required for types that can be directly
        allocated and returned from function calls.
        """
        return ScalarRef(element_type())

    def as_return_value(self, value: ScalarRef):
        """
        Convert the allocated return value into the value returned to the user when calling a kernel
        function. Default behaviour is just to return the value.
        """
        return value.value


register_python_type(ScalarRef,
                     ScalarRefMarshall(),
                     lambda stream, x: stream.write(type(x.value).__name + "\n"))


class ScalarDiffPair:
    def __init__(self, p: Optional[Any], d: Optional[Any], needs_grad: bool = True):
        super().__init__()
        self.primal = p if p is not None else 0.0
        self.grad = d if d is not None else 0.0
        self.needs_grad = needs_grad


class ScalarDiffPairMarshall(PythonMarshal):
    """
    Marshall for scalar ref (will be 1 per scalar element type)
    """

    def __init__(self):
        super().__init__(ScalarDiffPair)

    def get_element_shape(self, value: ScalarDiffPair):
        return get_python_type_marshall(value.primal).get_element_shape(value.primal)

    def get_element_type(self, value: ScalarDiffPair):
        return get_python_type_marshall(value.primal).get_element_type(value.primal)

    def is_writable(self, value: ScalarDiffPair) -> bool:
        return True

    def is_differentiable(self, value: ScalarDiffPair) -> bool:
        return value.needs_grad

    def primal_to_numpy(self, value: ScalarDiffPair):
        """
        For writing to a buffer, can just override the to_numpy function to get 
        the marshall for the value call its primal_to_numpy
        """
        return get_python_type_marshall(value.primal).primal_to_numpy(value.primal)

    def create_primal_calldata(self, device: Device, value: ScalarDiffPair, access: AccessType):
        """
        Return entry in call data for the primal value. For write, can rely on default
        buffer behaviour, but for reading need to return inner value
        """
        if access == AccessType.read:
            return value.primal
        else:
            return super().create_primal_calldata(device, value, access)

    def read_primal_calldata(self, device: Device, call_data: Any, access: AccessType, value: ScalarDiffPair):
        """
        For readback, need to use the value's marshal to convert numpy array, then
        store the result in the value field.
        """
        if access != AccessType.read:
            assert isinstance(call_data, Buffer)
            numpy_value = call_data.to_numpy()
            primal = get_python_type_marshall(value.primal).primal_from_numpy(numpy_value)
            value.primal = primal

    def derivative_to_numpy(self, value: ScalarDiffPair):
        """
        For writing to a buffer, can just override the to_numpy function to get 
        the marshall for the value call its derivative_to_numpy
        """
        return get_python_type_marshall(value.primal).primal_to_numpy(value.grad)

    def create_derivative_calldata(self, device: Device, value: ScalarDiffPair, access: AccessType):
        """
        Return entry in call data for the derivative value. For write, can rely on default
        buffer behaviour, but for reading need to return inner value
        """
        if access == AccessType.read:
            return value.grad
        else:
            return super().create_derivative_calldata(device, value, access)

    def read_derivative_calldata(self, device: Device, call_data: Any, access: AccessType, value: ScalarDiffPair):
        """
        For readback, need to use the value's marshal to convert numpy array, then
        store the result in the value field.
        """
        if access != AccessType.read:
            assert isinstance(call_data, Buffer)
            numpy_value = call_data.to_numpy()
            derivative = get_python_type_marshall(
                value.grad).primal_from_numpy(numpy_value)
            value.grad = derivative


register_python_type(ScalarDiffPair,
                     ScalarDiffPairMarshall(),
                     lambda stream, x: stream.write(type(x.value).__name + "\n"))


def intRef(init_value: int = 0) -> ScalarRef:
    return ScalarRef(int(init_value))


def floatRef(init_value: float = 0.0) -> ScalarRef:
    return ScalarRef(float(init_value))


def diffPair(
    p: Optional[Any] = None, d: Optional[Any] = None, needs_grad: bool = True
) -> ScalarDiffPair:
    return ScalarDiffPair(p, d, needs_grad)


def floatDiffPair(
    p: float = 0.0, d: float = 1.0, needs_grad: bool = True
) -> ScalarDiffPair:
    return diffPair(p, d, needs_grad)


def is_differentiable_buffer(val: Any):
    grad = getattr(val, "grad", None)
    if grad is None:
        return False
    needs_grad = getattr(val, "needs_grad", None)
    if needs_grad is None:
        return False
    if not isinstance(needs_grad, bool):
        return False
    return needs_grad


def to_numpy(buffer: Buffer):
    np = buffer.to_numpy()
    if isinstance(np, ndarray):
        return np
    else:
        raise ValueError("Buffer did not return an ndarray")
