# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import gc
import weakref

import pytest

import slangpy as spy
from slangpy.testing import helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_create_and_destroy_device_via_del(device_type: spy.DeviceType):
    device = helpers.get_device(device_type, use_cache=False)
    assert device is not None
    del device


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_create_and_destroy_device_via_none(device_type: spy.DeviceType):
    device = helpers.get_device(device_type, use_cache=False)
    assert device is not None
    device = None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_load_module_and_cleanup_in_order(device_type: spy.DeviceType):
    device = helpers.get_device(device_type, use_cache=False)
    assert device is not None

    module = device.load_module_from_source(
        module_name="module_from_source",
        source=r"""
        [shader("compute")]
        [numthreads(1, 1, 1)]
        void main() {
        }
    """,
    )

    module = None
    device = None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_load_module_and_cleanup_in_reverse_order(device_type: spy.DeviceType):
    device = helpers.get_device(device_type, use_cache=False)
    assert device is not None

    module = device.load_module_from_source(
        module_name="module_from_source",
        source=r"""
        [shader("compute")]
        [numthreads(1, 1, 1)]
        void main() {
        }
    """,
    )

    device = None
    module = None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_module_layout_does_not_keep_closed_device_alive(device_type: spy.DeviceType):
    created_device_count = len(spy.Device.get_created_devices())
    device = helpers.get_device(device_type, use_cache=False)
    module = device.load_module_from_source(
        module_name="module_with_cached_layout",
        source=r"""
        struct Foo {
            float value;
        };
        Foo foo;
    """,
    )
    layout = module.layout

    assert layout is not None

    device.close()
    layout = None
    module = None
    device = None
    gc.collect()

    assert len(spy.Device.get_created_devices()) == created_device_count


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_function_call_does_not_keep_closed_device_alive(device_type: spy.DeviceType):
    created_device_count = len(spy.Device.get_created_devices())
    device = helpers.get_device(device_type, use_cache=False)
    module = spy.Module.load_from_source(
        device,
        "module_with_reflection_caches",
        """
        struct Pair {
            float left;
            float right;
        };
        float add_pair(Pair value) { return value.left + value.right; }
        """,
    )

    assert float(module.add_pair({"left": 1.0, "right": 2.0})) == pytest.approx(3.0)

    device.close()
    module = None
    device = None
    gc.collect()

    assert len(spy.Device.get_created_devices()) == created_device_count


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_module_attribute_cache_does_not_create_ownership_cycle(device_type: spy.DeviceType):
    device = helpers.get_device(device_type, use_cache=False)
    module = spy.Module.load_from_source(
        device,
        "module_with_cached_attributes",
        """
        struct Pair {
            float left;
            float right;
        };
        float add_pair(Pair value) { return value.left + value.right; }
        """,
    )

    struct = module.Pair
    function = module.add_pair
    assert module.Pair is struct
    assert module.add_pair is function
    assert float(function({"left": 1.0, "right": 2.0})) == pytest.approx(3.0)

    call_data = function.debug_build_call_data({"left": 1.0, "right": 2.0})
    runtime = call_data.runtime
    module_ref = weakref.ref(module)
    struct_ref = weakref.ref(struct)
    function_ref = weakref.ref(function)
    call_data_ref = weakref.ref(call_data)
    runtime_ref = weakref.ref(runtime)

    device.close()
    del runtime, call_data, function, struct, module
    gc.collect()

    assert module_ref() is None
    assert struct_ref() is None
    assert function_ref() is None
    assert call_data_ref() is None
    assert runtime_ref() is None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_instance_method_cache_does_not_create_ownership_cycle(device_type: spy.DeviceType):
    device = helpers.get_device(device_type, use_cache=False)
    module = spy.Module.load_from_source(
        device,
        "module_with_cached_instance_methods",
        """
        struct Counter {
            int value;
            [mutating]
            void reset(int new_value) { value = new_value; }
        };
        """,
    )

    struct = module.Counter
    data = spy.Tensor.empty(device, dtype=struct, shape=(1,))
    instance = spy.InstanceList(struct, data)
    function = instance.reset
    assert instance.reset is function
    function(42)

    call_data = function.debug_build_call_data(42)
    runtime = call_data.runtime
    module_ref = weakref.ref(module)
    struct_ref = weakref.ref(struct)
    instance_ref = weakref.ref(instance)
    function_ref = weakref.ref(function)
    call_data_ref = weakref.ref(call_data)
    runtime_ref = weakref.ref(runtime)

    device.close()
    del runtime, call_data, function, instance, data, struct, module
    gc.collect()

    assert module_ref() is None
    assert struct_ref() is None
    assert instance_ref() is None
    assert function_ref() is None
    assert call_data_ref() is None
    assert runtime_ref() is None


def asserting_creation(device_type: spy.DeviceType):
    device = helpers.get_device(device_type, use_cache=False)
    assert device is not None

    module = device.load_module_from_source(
        module_name="module_from_source",
        source=r"""
        [shader("compute")]
        [numthreads(1, 1, 1)]
        void main() {
        }
    """,
    )
    raise Exception("Test failed")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_load_module_and_cleanup_through_assert(device_type: spy.DeviceType):
    with pytest.raises(Exception):
        asserting_creation(device_type)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
