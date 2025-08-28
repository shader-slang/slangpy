# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import slangpy as spy

pytest.skip("tev required for running these tests", allow_module_level=True)


def create_bitmap(
    width: int = 500,
    height: int = 500,
    component_type: spy.Bitmap.ComponentType = spy.Bitmap.ComponentType.float32,
):
    return spy.Bitmap(
        pixel_format=spy.Bitmap.PixelFormat.rgb,
        component_type=spy.Bitmap.ComponentType.float32,
        width=width,
        height=height,
    )


def test_show_in_tev():
    spy.tev.show(
        bitmap=create_bitmap(component_type=spy.Bitmap.ComponentType.float32),
        name="test1_float",
    )
    spy.tev.show(
        bitmap=create_bitmap(component_type=spy.Bitmap.ComponentType.uint8),
        name="test1_uint8",
    )
    spy.tev.show(
        bitmap=create_bitmap(component_type=spy.Bitmap.ComponentType.uint32),
        name="test1_uint32",
    )


def test_show_in_tev_async():
    spy.tev.show_async(bitmap=create_bitmap(), name="test2")


def test_show_in_tev_async_stress():
    for i in range(500):
        spy.tev.show_async(bitmap=create_bitmap(), name=f"test3_{i}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
