from typing import Any
import pytest
from kernelfunctions.backend import DeviceType, float2, float3
from kernelfunctions.function import Function
from kernelfunctions.instance import Instance
from kernelfunctions.module import Module
from kernelfunctions.struct import Struct
import kernelfunctions.tests.helpers as helpers
from kernelfunctions.types.buffer import NDBuffer
from kernelfunctions.types.valueref import ValueRef, floatRef
from kernelfunctions.utils import find_type_layout_for_buffer
import numpy as np


def load_test_module(device_type: DeviceType):
    device = helpers.get_device(device_type)
    return Module(device.load_module("test_modules.slang"))


class ThisType:
    def __init__(self, data: Any) -> None:
        super().__init__()
        self.data = data
        self.get_called = 0
        self.update_called = 0

    def get_this(self) -> Any:
        self.get_called += 1
        return self.data

    def update_this(self, value: Any) -> None:
        self.update_called += 1
        pass


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_this_interface(device_type: DeviceType):
    m = load_test_module(device_type)
    assert m is not None

    # Get particle struct
    Particle = m.Particle
    assert Particle is not None
    assert isinstance(Particle, Struct)

    # Get particle type so we can allocate a buffer
    particle_type_layout = find_type_layout_for_buffer(m.device_module.layout, "Particle")
    assert particle_type_layout is not None
    buffer = NDBuffer(m.device, particle_type_layout, 1)

    # Create a tiny wrapper around the buffer to provide the this interface
    this = ThisType(buffer)

    # Extend the Particle.reset function with the this interface and call it
    Particle_reset = Particle.reset.instance(this)
    Particle_reset(float2(1, 2), float2(3, 4))

    # Check the buffer has been correctly populated
    data = buffer.buffer.to_numpy().view(dtype=np.float32)
    assert len(data) == 11
    # position
    assert data[0] == 1.0
    assert data[1] == 2.0
    # direction
    assert data[2] == 3.0
    assert data[3] == 4.0

    # Check the this interface has been called
    # Get should have been called twice - once during setup, and once during call
    # Update should only have been called once during the call
    assert this.get_called == 2
    assert this.update_called == 1


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_this_interface_soa(device_type: DeviceType):
    m = load_test_module(device_type)
    assert m is not None

    # Get particle struct
    Particle = m.Particle
    assert Particle is not None
    assert isinstance(Particle, Struct)

    # Create a tiny wrapper around the buffer to provide the this interface
    this = ThisType({
        "position": NDBuffer(m.device, float2, 1),
        "velocity": float2(1, 0),
        "size": 0.5,
        "material": {
            "color": float3(1, 1, 1),
            "emission": float3(0, 0, 0)
        }
    })

    # Extend the Particle.reset function with the this interface and call it
    Particle_reset = Particle.reset.instance(this)
    Particle_reset(float2(1, 2), float2(3, 4))

    # Check the buffer has been correctly populated
    data = this.data['position'].buffer.to_numpy().view(dtype=np.float32)
    assert len(data) == 2
    assert data[0] == 1.0
    assert data[1] == 2.0


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_loose_instance_as_buffer(device_type: DeviceType):
    m = load_test_module(device_type)
    assert m is not None

    # Get particle struct
    Particle = m.Particle
    assert Particle is not None
    assert isinstance(Particle, Struct)

    # Get particle type so we can allocate a buffer
    particle_type_layout = find_type_layout_for_buffer(m.device_module.layout, "Particle")
    assert particle_type_layout is not None
    buffer = NDBuffer(m.device, particle_type_layout, 1)

    # Create a tiny wrapper around the buffer to provide the this interface
    instance = Instance(Particle, buffer)
    instance.construct(float2(1, 2), float2(3, 4))

    # Check the buffer has been correctly populated
    data = buffer.buffer.to_numpy().view(dtype=np.float32)
    assert len(data) == 11
    assert data[0] == 1.0
    assert data[1] == 2.0
    assert data[2] == 3.0
    assert data[3] == 4.0

    # Reset particle to be moving up
    instance.reset(float2(0, 0), float2(0, 1))

    # Update it once
    instance.update_position(1.0)

    # Check the buffer has been correctly updated
    data = buffer.buffer.to_numpy().view(dtype=np.float32)
    assert len(data) == 11
    assert data[0] == 0.0
    assert data[1] == 1.0
    assert data[2] == 0.0
    assert data[3] == 1.0


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_loose_instance_soa(device_type: DeviceType):
    m = load_test_module(device_type)
    assert m is not None

    # Get particle struct
    Particle = m.Particle
    assert Particle is not None
    assert isinstance(Particle, Struct)

    # Create a tiny wrapper around the buffer to provide the this interface
    instance = Instance(Particle, {
        "position": NDBuffer(m.device, float2, 1),
        "velocity": ValueRef(float2(9999)),
        "size": floatRef(9999),
        "material": {
            "color": ValueRef(float3(9999)),
            "emission": ValueRef(float3(9999))
        }
    })

    instance.construct(float2(1, 2), float2(3, 4))

    # Check the buffer has been correctly populated
    data = instance.position.buffer.to_numpy().view(dtype=np.float32)
    assert data[0] == 1.0
    assert data[1] == 2.0

    # Reset particle to be moving up
    instance.reset(float2(0, 0), float2(0, 1))

    # Update it once
    instance.update_position(1.0)

    # Check the buffer has been correctly updated
    data = instance.position.buffer.to_numpy().view(dtype=np.float32)
    assert len(data) == 2
    assert data[0] == 0.0
    assert data[1] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
