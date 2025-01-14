# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from slangpy.core.function import Function
from slangpy.core.struct import Struct
from slangpy.core.utils import find_type_layout_for_buffer

import slangpy.tests.helpers as helpers
from slangpy import Module
from slangpy.backend import DeviceType, float2, float3
from slangpy.types.buffer import NDBuffer


def load_test_module(device_type: DeviceType):
    device = helpers.get_device(device_type)
    return Module(device.load_module("test_modules.slang"))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_create_module(device_type: DeviceType):
    m = load_test_module(device_type)
    assert m is not None

    Particle = m.Particle
    assert Particle is not None
    assert isinstance(Particle, Struct)

    # TODO: Want to make material child of struct but not available at current in slang
    Material = m.Material
    # Material = Particle.Material
    assert isinstance(Material, Struct)

    particle_dot__init__ = Particle.__init
    assert isinstance(particle_dot__init__, Function)

    particle_dot_reset = Particle.reset
    assert isinstance(particle_dot_reset, Function)

    particle_dot_calc_next_position = Particle.calc_next_position
    assert isinstance(particle_dot_calc_next_position, Function)

    particle_dot_update_position = Particle.update_position
    assert isinstance(particle_dot_update_position, Function)

    particle_dot_material_dot__init__ = Material.__init
    assert isinstance(particle_dot_material_dot__init__, Function)

    get_particle_quad = m.get_particle_quad
    assert isinstance(get_particle_quad, Function)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_init_particle(device_type: DeviceType):
    m = load_test_module(device_type)
    Particle = m.Particle
    assert isinstance(Particle, Struct)

    buffer = NDBuffer(m.device, Particle, 1)

    # Call constructor, which returns particles
    Particle.__init(float2(1.0, 2.0), float2(3.0, 4.0), _result=buffer)

    data = buffer.storage.to_numpy().view(dtype=np.float32)
    assert len(data) == 11
    # position
    assert data[0] == 1.0
    assert data[1] == 2.0
    # direction
    assert data[2] == 3.0
    assert data[3] == 4.0
    # size
    assert data[4] == 0.5
    # mat.color
    assert data[5] == 1
    assert data[6] == 1
    assert data[7] == 1
    # mat.emission
    assert data[8] == 0
    assert data[9] == 0
    assert data[10] == 0


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_read_only_func(device_type: DeviceType):
    m = load_test_module(device_type)
    Particle = m.Particle
    assert isinstance(Particle, Struct)

    # Create and init a buffer of particles
    buffer = NDBuffer(m.device, Particle, 1)
    Particle.__init(float2(0, 0), float2(0.1, 0.2), _result=buffer)

    # Get next position of all particles
    next_pos = Particle.calc_next_position(buffer, 0.5)

    data = next_pos.storage.to_numpy().view(dtype=np.float32)
    assert np.allclose(data, [0.05, 0.1])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_mutable_func(device_type: DeviceType):
    m = load_test_module(device_type)
    Particle = m.Particle
    assert isinstance(Particle, Struct)

    # Create and init a buffer of particles
    buffer = NDBuffer(m.device, Particle, 1)
    Particle.__init(float2(0, 0), float2(0.1, 0.2), _result=buffer)

    # Update position of all particles
    Particle.update_position(buffer, 0.5)

    data = buffer.storage.to_numpy().view(dtype=np.float32)
    assert np.allclose(data[:2], [0.05, 0.1])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_read_back_with_global_func(device_type: DeviceType):
    m = load_test_module(device_type)
    Particle = m.Particle
    assert isinstance(Particle, Struct)

    # Create and init a buffer of particles
    buffer = NDBuffer(m.device, Particle, 1)
    Particle.__init(float2(0, 0), float2(0, 0.2), _result=buffer)

    # Update position of particles
    Particle.update_position(buffer, 0.5)

    # Get quad of all particles
    quad_type_layout = find_type_layout_for_buffer(m.device_module.layout, "float2[4]")
    results = NDBuffer(m.device, quad_type_layout, 1)
    m.get_particle_quad(buffer, _result=results)

    data = results.storage.to_numpy().view(dtype=np.float32).reshape(-1, 2)
    assert np.allclose(data, [[-0.5, 0.6], [0.5, 0.6], [0.5, -0.4], [-0.5, -0.4]])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_soa_particles(device_type: DeviceType):
    m = load_test_module(device_type)
    Particle = m.Particle
    assert isinstance(Particle, Struct)

    soa_particles = {
        "position": NDBuffer(m.device, float2, 1),
        "velocity": float2(1, 0),
        "size": 0.5,
        "material": {
            "color": float3(1, 1, 1),
            "emission": float3(0, 0, 0)
        }
    }

    # Create and init a buffer of particles
    Particle.__init(float2(0, 0), float2(0.1, 0.2), _result=soa_particles)

    # Update position of all particles
    Particle.update_position(soa_particles, 0.5)

    data = soa_particles['position'].storage.to_numpy().view(dtype=np.float32)
    assert np.allclose(data, [0.5, 0])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
