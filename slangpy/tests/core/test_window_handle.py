# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys

import pytest

import slangpy as spy


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_window_handle_hwnd() -> None:
    assert spy.WindowHandle.from_hwnd(0x1234).type == spy.WindowHandleType.hwnd
    assert spy.WindowHandle(0x1234).type == spy.WindowHandleType.hwnd


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
def test_window_handle_nswindow() -> None:
    assert spy.WindowHandle.from_nswindow(0x1234).type == spy.WindowHandleType.nswindow
    assert spy.WindowHandle(0x1234).type == spy.WindowHandleType.nswindow


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux only")
def test_window_handle_linux() -> None:
    xlib = spy.WindowHandle.from_xlib_window(display=0x1234, window=42)
    assert xlib.type == spy.WindowHandleType.xlib_window
    assert spy.WindowHandle(0x1234, 42).type == spy.WindowHandleType.xlib_window
    wayland = spy.WindowHandle.from_wayland_surface(display=0x1234, surface=0x5678)
    assert wayland.type == spy.WindowHandleType.wayland_surface


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
