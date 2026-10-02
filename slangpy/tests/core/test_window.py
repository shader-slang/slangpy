# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import pytest
import slangpy as spy


def test_window_minimized_api_surface():
    assert hasattr(spy.Window, "is_minimized")
    assert hasattr(spy.Window, "on_iconify")


@pytest.mark.skipif(
    not os.environ.get("DISPLAY"),
    reason="GLFW window creation requires a display server",
)
def test_window_minimized_cache_and_callback():
    window = spy.Window(width=64, height=64, title="test")
    assert window.is_minimized() is False
    assert window.on_iconify is None

    def on_iconify(minimized: bool) -> None:
        pass

    window.on_iconify = on_iconify
    assert window.on_iconify is not None
