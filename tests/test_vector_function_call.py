import pytest
import sys
from pathlib import Path
import sgl

sys.path.append(str(Path(__file__).parent))
import helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_function(device_type: sgl.DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "add_numbers",
        r"""
void add_numbers(float2 a, float2 b) {
}
""",
    )

    function(sgl.float2(1, 2), sgl.float2(2, 3))
    function(1.0, sgl.float2(2, 3))
    function(sgl.float2(2, 3), 2.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
