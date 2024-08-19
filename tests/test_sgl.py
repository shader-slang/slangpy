import pytest
import sys
import sgl
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
import helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_sgl(device_type: sgl.DeviceType):
    device = helpers.get_device(device_type)
    assert device.desc.type == device_type
    assert device.desc.enable_debug_layers == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
