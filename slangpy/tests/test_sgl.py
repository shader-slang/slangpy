# SPDX-License-Identifier: Apache-2.0
import pytest

import slangpy.tests.helpers as helpers
from slangpy.backend import DeviceType


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_sgl(device_type: DeviceType):
    device = helpers.get_device(device_type)
    assert device.desc.type == device_type


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
