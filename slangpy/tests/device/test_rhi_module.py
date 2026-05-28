# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

import slangpy as spy
from slangpy.testing import helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_rhi_module(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    device.load_module("test_rhi_module.slang")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
