# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

import slangpy as spy
from slangpy.native_func import BaseModule, BaseStruct
from slangpy.native_refl import Layout
from slangpy.testing import helpers


MODULE_SOURCE = """
struct Foo {
    int value;
};
"""


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_module_and_struct_have_native_bridge(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE_SOURCE)

    assert isinstance(module, BaseModule)
    assert module.layout.find_type_by_name("Foo") is not None

    struct = module.Foo.as_struct()

    assert isinstance(struct, BaseStruct)

    tensor = spy.Tensor.empty(device, (2,), dtype=struct)
    assert tensor.dtype is struct.struct


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_native_refl_layout_tracks_hot_reload_generation(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE_SOURCE)
    layout = Layout(module.module.layout)

    generation = layout.generation
    layout.on_hot_reload(module.module.layout)
    module.on_hot_reload()

    assert layout.generation == generation + 1
    assert module.layout.find_type_by_name("Foo") is not None
