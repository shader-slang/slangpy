# SPDX-License-Identifier: Apache-2.0
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from sgl import Buffer, Device

from slangpy.core.struct import Struct

import slangpy.tests.helpers as helpers
from slangpy import (InstanceList, InstanceBuffer,
                     Module)
from slangpy.backend import DeviceType, float2, float3, math
from slangpy.types import NDBuffer, Tensor
from slangpy.types.randfloatarg import RandFloatArg
from slangpy.types.valueref import ValueRef, floatRef
from slangpy.experimental.diffinstancelist import InstanceDifferentiableBuffer


def load_module(device_type: DeviceType, name: str = "test_modules.slang") -> Module:
    device = helpers.get_device(device_type)
    return Module(device.load_module(name))


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


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_this_interface(device_type: DeviceType):
    m = load_module(device_type)
    assert m is not None

    # Get particle struct
    Particle = m.Particle
    assert Particle is not None
    assert isinstance(Particle, Struct)

    # Allocate a buffer
    buffer = NDBuffer(m.device, Particle, 1)

    # Create a tiny wrapper around the buffer to provide the this interface
    this = ThisType(buffer)

    # Extend the Particle.reset function with the this interface and call it
    Particle_reset = Particle.reset.bind(this)
    Particle_reset(float2(1, 2), float2(3, 4))

    # Check the buffer has been correctly populated
    data = buffer.storage.to_numpy().view(dtype=np.float32)
    assert len(data) == 11
    # position
    assert data[0] == 1.0
    assert data[1] == 2.0
    # direction
    assert data[2] == 3.0
    assert data[3] == 4.0

    # Check the this interface has been called
    # Get should have been called 3 times - once for hash, once during setup, and once during call
    # Update should only have been called once during the call
    assert this.get_called == 3
    assert this.update_called == 1


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_this_interface_soa(device_type: DeviceType):
    m = load_module(device_type)
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
    Particle_reset = Particle.reset.bind(this)
    Particle_reset(float2(1, 2), float2(3, 4))

    # Check the buffer has been correctly populated
    data = this.data['position'].storage.to_numpy().view(dtype=np.float32)
    assert len(data) == 2
    assert data[0] == 1.0
    assert data[1] == 2.0


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_loose_instance_as_buffer(device_type: DeviceType):
    m = load_module(device_type)
    assert m is not None

    # Get particle struct
    Particle = m.Particle
    assert Particle is not None
    assert isinstance(Particle, Struct)

    # Sllocate a buffer
    buffer = NDBuffer(m.device, Particle, 1)

    # Create a tiny wrapper around the buffer to provide the this interface
    instance = InstanceList(Particle, buffer)
    instance.construct(float2(1, 2), float2(3, 4))

    # Check the buffer has been correctly populated
    data = buffer.storage.to_numpy().view(dtype=np.float32)
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
    data = buffer.storage.to_numpy().view(dtype=np.float32)
    assert len(data) == 11
    assert data[0] == 0.0
    assert data[1] == 1.0
    assert data[2] == 0.0
    assert data[3] == 1.0


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_loose_instance_soa(device_type: DeviceType):
    m = load_module(device_type)
    assert m is not None

    # Get particle struct
    Particle = m.Particle
    assert Particle is not None
    assert isinstance(Particle, Struct)

    # Create a tiny wrapper around the buffer to provide the this interface
    instance = InstanceList(Particle, {
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
    data = instance.position.storage.to_numpy().view(dtype=np.float32)
    assert data[0] == 1.0
    assert data[1] == 2.0

    # Reset particle to be moving up
    instance.reset(float2(0, 0), float2(0, 1))

    # Update it once
    instance.update_position(1.0)

    # Check the buffer has been correctly updated
    data = instance.position.storage.to_numpy().view(dtype=np.float32)
    assert len(data) == 2
    assert data[0] == 0.0
    assert data[1] == 1.0


def particle_update_positions(data: npt.NDArray[np.float32], dt: float):
    for i in range(0, len(data)):
        data[i][0] += data[i][2] * dt
        data[i][1] += data[i][3] * dt


def get_particle_quads(data: npt.NDArray[np.float32]):
    results = np.ndarray((len(data), 4, 2), dtype=np.float32)
    for i in range(0, len(data)):
        pos = float2(data[i][0:2])
        vel = float2(data[i][2:4])
        size = data[i][4]
        if (math.length(vel) < 0.01):
            vel = float2(0, 1)

        up = math.normalize(vel)
        right = float2(-up.y, up.x)
        center = pos

        quad = [
            center + (right + up) * size,
            center + (-right + up) * size,
            center + (-right - up) * size,
            center + (right - up) * size
        ]
        for vertex in range(0, 4):
            results[i][vertex] = [quad[vertex].x, quad[vertex].y]
    return results


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pass_instance_to_function(device_type: DeviceType):
    # Use test system helper to load a slangpy module from a file
    m = load_module(device_type, "test_modules.slang")
    assert m is not None

    # Get particle struct
    Particle = m.Particle
    assert isinstance(Particle, Struct)

    # Create storage for particles in a simple buffer
    particles = InstanceBuffer(Particle, shape=(1000,))

    # Call the slang constructor on all particles in the buffer,
    # assigning each a constant starting position and a random velocity
    particles.construct(position=float2(10, 10),
                        velocity=RandFloatArg(min=-1, max=1, dim=2))
    expected_particles = particles._data.storage.to_numpy().view(dtype=np.float32).reshape(-1, 11)

    # Call the slang function 'Particle::update_position' to update them
    # and do the same for the python version
    particles.update_position(dt=1.0/60.0)
    particle_update_positions(expected_particles, 1.0/60.0)

    # Check the numpy buffer and the slang buffer are the same
    particle_data = particles._data.storage.to_numpy().view(dtype=np.float32).reshape(-1, 11)
    assert np.allclose(particle_data, expected_particles)

    # Define a 'Quad' type which is just an array of float2s, and make a buffer for them
    Quad = m['float2[4]']
    assert isinstance(Quad, Struct)
    quads = InstanceBuffer(Quad, particles.shape)

    # Call the slang function 'get_particle_quad' which takes particles and returns quads
    m.get_particle_quad(particles, _result=quads)
    expected_quads = get_particle_quads(expected_particles)

    # Read out all the quads as numpy arrays of floats
    quad_data = quads._data.storage.to_numpy().view(dtype=np.float32).reshape(-1, 4, 2)
    assert np.allclose(quad_data, expected_quads)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pass_nested_instance_to_function(device_type: DeviceType):
    # Use test system helper to load a slangpy module from a file
    m = load_module(device_type, "test_modules.slang")
    assert m is not None

    # Get particle and material structs
    Particle = m.Particle
    assert isinstance(Particle, Struct)

    # Get particle struct
    Material = m.Material
    assert isinstance(Material, Struct)

    # Create storage for particles in a simple buffer
    particles = InstanceList(Particle, {
        'position': NDBuffer(m.device, float2, 1000),
        'velocity': float2(0, 0),
        'size': 0.5,
        'material': InstanceBuffer(Material, shape=(1000,))
    })

    # Call the slang constructor on all particles in the buffer,
    # assigning each a constant starting position and a random velocity
    particles.construct(position=float2(10, 10), velocity=float2(0, 0))

    # Check colors are white and emission is black!
    material_data = particles._data['material']._data.storage.to_numpy().view(
        dtype=np.float32).reshape(-1, 6)
    for i in range(0, len(material_data)):
        material_data[i][0] = 1
        material_data[i][1] = 1
        material_data[i][2] = 1
        material_data[i][3] = 0
        material_data[i][4] = 0
        material_data[i][5] = 0


class CustomInstanceList:
    def __init__(self, device: Device, data: list[float2]):
        super().__init__()
        self.device = device
        self.data = data

    def get_this(self) -> Any:
        buffer = NDBuffer(self.device, float2, len(self.data))
        np_data = np.array([[v.x, v.y] for v in self.data], dtype=np.float32)
        buffer.from_numpy(np_data)
        return buffer

    def update_this(self, value: Buffer) -> None:
        np_data = value.to_numpy().view(dtype=np.float32).reshape(-1, 2)
        self.data = [float2(v[0], v[1]) for v in np_data]


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_custom_instance_list(device_type: DeviceType):
    # Use test system helper to load a slangpy module from a file
    m = load_module(device_type, "test_modules.slang")
    assert m is not None

    # Get particle and material structs
    Particle = m.Particle
    assert isinstance(Particle, Struct)

    # Create storage for particles in a simple buffer
    particles = InstanceList(Particle, {
        'position': CustomInstanceList(m.device, [float2(x, x) for x in [1, 2, 3, 4]]),
    })

    # Call the slang constructor on all particles in the buffer,
    # assigning each a constant starting position and a random velocity
    particles.construct(position=float2(10, 10), velocity=float2(0, 0))

    # Get particl data
    data = particles._data['position'].data
    for i in range(0, len(data)):
        assert data[i] == 10


class ExtendedInstanceList(InstanceList):
    def __init__(self, struct: Struct):
        super().__init__(struct)
        self.position = NDBuffer(struct.device, float2, 1000)
        self.velocity = NDBuffer(struct.device, float2, 1000)
        self.size = 0.5
        self.material = {
            'color': float3(1, 1, 1),
            'emission': float3(0, 0, 0)
        }

    def update(self):
        self.update_position(1.0/60.0)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_extended_instance_list(device_type: DeviceType):
    # Use test system helper to load a slangpy module from a file
    m = load_module(device_type, "test_modules.slang")
    assert m is not None

    # Get particle and material structs
    Particle = m.Particle
    assert isinstance(Particle, Struct)

    # Create storage for particles in a simple buffer
    particles = ExtendedInstanceList(Particle)

    # Call the slang constructor on all particles in the buffer,
    # assigning each a constant starting position and a random velocity
    particles.construct(position=float2(10, 10), velocity=float2(0, 1))

    # Call custom update function which internally calls update_position
    particles.update()

    # Check the buffer has been correctly updated
    data = particles.position.to_numpy().view(dtype=np.float32).reshape(-1, 2)
    assert np.allclose(data, [10, 10+1.0/60.0])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_backwards_diff(device_type: DeviceType):
    # Use test system helper to load a slangpy module from a file
    m = load_module(device_type, "test_modules.slang")
    assert m is not None

    # Get particle struct
    Particle = m.Particle
    assert isinstance(Particle, Struct)

    # Create storage for particles in a simple buffer
    particles = InstanceDifferentiableBuffer(Particle, shape=(1000,))

    # Call the slang constructor on all particles in the buffer,
    # assigning each a constant starting position and a random velocity
    particles.construct(position=float2(10, 10),
                        velocity=RandFloatArg(min=-1, max=1, dim=2))

    # Get next position of particles (automatically returns differentiable buffer of correct size)
    next_positions = particles.calc_next_position(dt=1.0/60.0, _result='tensor')
    next_positions = next_positions.with_grads()

    # Init the gradients of next positions to 1 for the backwards pass
    next_positions.grad.storage.from_numpy(np.ones((1000, 2), dtype=np.float32))

    # Make a buffer of 1000 identical dts, so we can get back the unique grads for each one
    dts = Tensor.empty(m.device, shape=(1000,), dtype=float).with_grads()
    dts.storage.from_numpy(np.full((1000,), 1.0/60.0, dtype=np.float32))

    # Backwards pass
    particles.calc_next_position.bwds(dt=dts, _result=next_positions)

    # Read back all primals and gradients we ended up with into numpy arrays
    particle_primals = particles.primal_to_numpy().view(dtype=np.float32).reshape(-1, 11)
    particle_grads = particles.grad_to_numpy().view(dtype=np.float32).reshape(-1, 11)
    dt_grads = dts.grad.storage.to_numpy().view(dtype=np.float32)
    next_positions = next_positions.storage.to_numpy().view(dtype=np.float32).reshape(-1, 2)

    for particle_idx in range(0, len(particle_primals)):
        pos = particle_primals[particle_idx][0:2]
        vel = particle_primals[particle_idx][2:4]
        next_pos = next_positions[particle_idx]

        # Expect particle to have moved by 1/60th of its velocity
        # q = p + vel * dt
        expected_pos = pos + vel * 1.0/60.0
        assert np.allclose(next_pos, expected_pos)

        # dq/dp = 10
        pos_grad = particle_grads[particle_idx][0:2]
        assert np.allclose(pos_grad, [1, 1])

        # dq/dv = dt
        vel_grad = particle_grads[particle_idx][2:4]
        assert np.allclose(vel_grad, [1.0/60.0, 1.0/60.0])

        # dq/ddt = v
        dt_grad = dt_grads[particle_idx]
        assert np.allclose(dt_grad, vel[0]+vel[1])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
