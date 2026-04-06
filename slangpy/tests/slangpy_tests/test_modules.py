# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np

from slangpy import DeviceType, float2, float3, Module
from slangpy.core.function import Function
from slangpy.core.struct import Struct
from slangpy.core.utils import find_type_layout_for_buffer
from slangpy.types import Tensor
from slangpy.testing import helpers


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

    buffer = Tensor.empty(m.device, dtype=Particle, shape=(1,))

    # Call constructor, which returns particles
    Particle.__init(float2(1.0, 2.0), float2(3.0, 4.0), _result=buffer)

    data = helpers.read_tensor_from_numpy(buffer)
    print(data)

    # position
    positions = np.array([item["position"] for item in data])
    assert np.allclose(positions, [1.0, 2.0])
    # velocity
    velocity = np.array([item["velocity"] for item in data])
    assert np.allclose(velocity, [3.0, 4.0])
    # size
    sizes = np.array([item["size"] for item in data])
    assert np.allclose(sizes, [0.5])
    # mat.color
    colors = np.array([item["material"]["color"] for item in data])
    assert np.allclose(colors, [1.0, 1.0, 1.0])
    # mat.emission
    emissions = np.array([item["material"]["emission"] for item in data])
    assert np.allclose(emissions, [0.0, 0.0, 0.0])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_read_only_func(device_type: DeviceType):
    m = load_test_module(device_type)
    Particle = m.Particle
    assert isinstance(Particle, Struct)

    # Create and init a buffer of particles
    buffer = Tensor.empty(m.device, dtype=Particle, shape=(1,))
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
    buffer = Tensor.empty(m.device, dtype=Particle, shape=(1,))
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
    buffer = Tensor.empty(m.device, dtype=Particle, shape=(1,))
    Particle.__init(float2(0, 0), float2(0, 0.2), _result=buffer)

    # Update position of particles
    Particle.update_position(buffer, 0.5)

    # Get quad of all particles
    quad_type_layout = find_type_layout_for_buffer(m.device_module.layout, "float2[4]")
    results = Tensor.empty(m.device, dtype=quad_type_layout, shape=(1,))
    m.get_particle_quad(buffer, _result=results)

    data = results.storage.to_numpy().view(dtype=np.float32).reshape(-1, 2)
    assert np.allclose(data, [[-0.5, 0.6], [0.5, 0.6], [0.5, -0.4], [-0.5, -0.4]])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_soa_particles(device_type: DeviceType):
    m = load_test_module(device_type)
    Particle = m.Particle
    assert isinstance(Particle, Struct)

    soa_particles = {
        "position": Tensor.empty(m.device, dtype=float2, shape=(1,)),
        "velocity": float2(1, 0),
        "size": 0.5,
        "material": {"color": float3(1, 1, 1), "emission": float3(0, 0, 0)},
    }

    # Create and init a buffer of particles
    Particle.__init(float2(0, 0), float2(0.1, 0.2), _result=soa_particles)

    # Update position of all particles
    Particle.update_position(soa_particles, 0.5)

    data = soa_particles["position"].storage.to_numpy().view(dtype=np.float32)
    assert np.allclose(data, [0.5, 0])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_load_from_source(device_type: DeviceType):
    """Module.load_from_source should create a working module from a source string."""
    device = helpers.get_device(device_type)
    m = Module.load_from_source(device, "test_inline", "float add1(float x) { return x + 1; }")
    result = m.add1(2.0)
    assert abs(float(result) - 3.0) < 1e-5


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_module_property(device_type: DeviceType):
    """Module.module should return the underlying SlangModule."""
    m = load_test_module(device_type)
    assert m.module is m.device_module


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_require_struct_not_found(device_type: DeviceType):
    """require_struct should raise ValueError for nonexistent struct."""
    m = load_test_module(device_type)
    with pytest.raises(ValueError, match="Could not find struct"):
        m.require_struct("NonexistentStruct")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_find_function_in_struct(device_type: DeviceType):
    """find_function_in_struct should find methods by struct name or object."""
    m = load_test_module(device_type)
    func = m.find_function_in_struct("Particle", "calc_next_position")
    assert func is not None
    assert isinstance(func, Function)

    func_none = m.find_function_in_struct("Particle", "nonexistent_method")
    assert func_none is None

    func_bad_struct = m.find_function_in_struct("NoSuchStruct", "calc_next_position")
    assert func_bad_struct is None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_module_repr(device_type: DeviceType):
    """Module.__repr__ should return a useful string."""
    m = load_test_module(device_type)
    r = repr(m)
    assert "Module(" in r
    assert "test_modules" in r


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_module_getitem(device_type: DeviceType):
    """Module['name'] should work like module.name."""
    m = load_test_module(device_type)
    func = m["add_vectors"]
    assert isinstance(func, Function)

    with pytest.raises(AttributeError, match="no function or type"):
        m["nonexistent_thing"]


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tensor_empty_with_slangtype(device_type: DeviceType):
    """Tensor.empty with a pre-resolved SlangType exercises resolve_element_type SlangType path."""
    m = load_test_module(device_type)
    float_type = m.layout.find_type_by_name("float")
    tensor = Tensor.empty(m.device, (4,), dtype=float_type, program_layout=m.layout)
    assert tensor.shape[0] == 4


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tensor_empty_with_struct(device_type: DeviceType):
    """Tensor.empty with a Struct dtype exercises resolve_element_type Struct path."""
    m = load_test_module(device_type)
    Material = m.Material
    tensor = Tensor.empty(m.device, (2,), dtype=Material)
    assert tensor.shape[0] == 2


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_slang_to_numpy_unsupported(device_type: DeviceType):
    """slang_to_numpy should return None for non-scalar types like structs."""
    from slangpy.reflection.lookup import slang_to_numpy
    m = load_test_module(device_type)
    struct_type = m.layout.find_type_by_name("Material")
    assert slang_to_numpy(struct_type) is None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_numpy_to_slang_unsupported(device_type: DeviceType):
    """numpy_to_slang should return None for unsupported numpy dtypes."""
    from slangpy.reflection.lookup import numpy_to_slang
    device = helpers.get_device(device_type)
    result = numpy_to_slang(np.dtype("complex128"), device, None)
    assert result is None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_function_set_invalid(device_type: DeviceType):
    """FunctionNode.set() with a non-dict/non-callable arg should raise ValueError."""
    m = load_test_module(device_type)
    func = m.add_vectors.as_func()
    with pytest.raises(ValueError, match="Set requires"):
        func.set(42)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_function_write_non_callable(device_type: DeviceType):
    """FunctionNode.write() with non-callable should raise ValueError."""
    m = load_test_module(device_type)
    func = m.add_vectors.as_func()
    with pytest.raises(ValueError, match="callable"):
        func.write(42)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_function_return_type_bad_string(device_type: DeviceType):
    """FunctionNode.return_type() with unknown string should raise ValueError."""
    m = load_test_module(device_type)
    func = m.add_vectors.as_func()
    with pytest.raises(ValueError, match="Unknown return type"):
        func.return_type("bad_type")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_function_as_struct(device_type: DeviceType):
    """FunctionNode.as_struct() should raise ValueError."""
    m = load_test_module(device_type)
    func = m.add_vectors.as_func()
    with pytest.raises(ValueError, match="Cannot convert"):
        func.as_struct()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
