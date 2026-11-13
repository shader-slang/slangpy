# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable, Iterator

import pytest
import slangpy as spy


@pytest.fixture(scope="module")
def desktop() -> None:
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        pytest.skip("Window tests require an X11 desktop")
    try:
        monitors = spy.platform.get_monitors()
    except RuntimeError as error:
        if "built without GLFW" in str(error):
            pytest.skip("SlangPy was built without window support")
        raise
    assert monitors


def _position() -> spy.int2:
    work = next(m.work_area for m in spy.platform.get_monitors() if m.primary)
    return spy.int2(work.x + 80, work.y + 80)


@pytest.fixture
def window(desktop: None) -> Iterator[spy.Window]:
    value = spy.Window(width=400, height=300, position=_position(), visible=False)
    try:
        yield value
    finally:
        value.hide()
        value.close()


def _wait(window: spy.Window, condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        window.process_events()
        if condition():
            return
        time.sleep(0.01)
    assert condition(), "Window manager did not apply the requested state"


def _probe_monitor_lifetime() -> None:
    # Run in a fresh process to catch initialization hidden by another test's window.
    before = spy.platform.get_monitors()
    assert before and sum(m.primary for m in before) == 1
    saved_bounds = spy.int4(before[0].bounds)
    for monitor in before:
        assert monitor.name
        assert monitor.identifier is None or isinstance(monitor.identifier, str)
        assert monitor.bounds.z > 0 and monitor.bounds.w > 0
        assert monitor.work_area.z > 0 and monitor.work_area.w > 0
        assert monitor.content_scale.x > 0 and monitor.content_scale.y > 0

    for _ in range(2):
        first = spy.Window(width=320, height=240, visible=False)
        second = spy.Window(width=320, height=240, visible=False)
        assert spy.platform.get_monitors()
        assert spy.platform.display_scale_factor() > 0
        del first
        gc.collect()
        # Destroying the first window and querying monitors must leave this alive.
        second.resize(360, 260)
        assert second.size == spy.uint2(360, 260)
        assert spy.platform.get_monitors()
        del second
        gc.collect()
        assert before[0].bounds == saved_bounds
        assert spy.platform.get_monitors()


def test_monitor_lifetime_before_and_after_windows(desktop: None) -> None:
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--probe-monitors"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("mode", ["import", "query", "windows"])
def test_glfw_shutdown(desktop: None, mode: str) -> None:
    # Keep wrappers and callback cycles alive until interpreter teardown.
    # Exercise shutdown both with and without initialized GLFW resources.
    script = textwrap.dedent(
        """\
        import sys
        import slangpy as spy

        if sys.argv[1] != "import":
            assert spy.platform.get_monitors()
        windows = []
        if sys.argv[1] == "windows":
            for _ in range(2):
                window = spy.Window(width=320, height=240, visible=False)
                window.cursor_shape = spy.CursorShape.hand
                # Retain a callback cycle as well as ordinary Python references.
                def on_resize(width: int, height: int, owner: spy.Window = window) -> None:
                    owner.title = f"{width}x{height}"
                window.on_resize = on_resize
                windows.append(window)
            del window, on_resize
        print("GLFW shutdown probe ready", flush=True)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, mode],
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "GLFW shutdown probe ready" in result.stdout, output
    assert "GLFW error" not in output, output
    assert "Exception ignored" not in output, output


def test_hidden_placement_and_window_metrics(window: spy.Window) -> None:
    assert not window.is_visible()
    assert not window.is_minimized()
    assert not window.is_maximized()
    assert not window.is_fullscreen()
    assert window.size == spy.uint2(400, 300)
    assert window.position == _position()
    placement = window.placement
    assert placement.normal_size == window.size
    assert placement.normal_position == window.position
    assert not placement.maximized
    assert window.content_scale.x > 0 and window.content_scale.y > 0
    assert all(window.frame_size[i] >= 0 for i in range(4))

    window.resize(420, 320)
    window.position = _position() + spy.int2(20, 10)
    _wait(window, lambda: window.size == spy.uint2(420, 320))
    assert window.placement.normal_size == spy.uint2(420, 320)
    assert window.placement.normal_position == _position() + spy.int2(20, 10)
    # Previously returned snapshots do not track the live window.
    assert placement.normal_size == spy.uint2(400, 300)


def test_show_hide(window: spy.Window) -> None:
    window.show()
    _wait(window, lambda: window.is_visible())
    window.hide()
    _wait(window, lambda: not window.is_visible())
    assert not window.is_minimized()


@pytest.mark.parametrize("visible", [False, True])
@pytest.mark.parametrize(
    "mode",
    [
        spy.WindowMode.normal,
        spy.WindowMode.minimized,
        spy.WindowMode.maximized,
        spy.WindowMode.fullscreen,
    ],
)
def test_initial_window_state(desktop: None, mode: spy.WindowMode, visible: bool) -> None:
    window = spy.Window(width=400, height=300, position=_position(), mode=mode, visible=visible)
    try:
        assert window.is_fullscreen() == (mode == spy.WindowMode.fullscreen)
        if not visible:
            assert not window.is_visible()
            assert not window.is_minimized()
            assert not window.is_maximized()
            window.show()
            _wait(window, lambda: window.is_visible())
            assert not window.is_minimized()
            assert not window.is_maximized()
        elif mode == spy.WindowMode.minimized:
            _wait(window, lambda: window.is_minimized())
        elif mode == spy.WindowMode.maximized:
            _wait(window, lambda: window.is_maximized())
        else:
            _wait(window, lambda: window.is_visible())

        if mode != spy.WindowMode.fullscreen:
            assert window.placement.normal_size == spy.uint2(400, 300)
            assert window.placement.normal_position == _position()
            window.restore()
            _wait(window, lambda: not window.is_minimized() and not window.is_maximized())
            assert window.size == spy.uint2(400, 300)
            assert window.position == _position()
    finally:
        window.hide()
        window.close()


def test_maximize_minimize_restore(window: spy.Window) -> None:
    window.show()
    _wait(window, lambda: window.is_visible())
    original = window.placement.to_dict()
    window.maximize()
    _wait(window, lambda: window.is_maximized())
    window.minimize()
    _wait(window, lambda: window.is_minimized())
    assert window.placement.to_dict() == {**original, "maximized": True}

    window.restore()
    _wait(window, lambda: not window.is_minimized())
    # The window system may restore to maximized first.
    if window.is_maximized():
        window.restore()
    _wait(window, lambda: not window.is_maximized())
    assert window.placement.to_dict() == original


@pytest.mark.parametrize("maximized", [False, True])
def test_reopen_placement(window: spy.Window, maximized: bool) -> None:
    window.position = _position() + spy.int2(40, 60)
    window.size = spy.uint2(520, 360)
    window.show()
    _wait(window, lambda: window.is_visible())
    if maximized:
        window.maximize()
        _wait(window, lambda: window.is_maximized())
    window.minimize()
    _wait(window, lambda: window.is_minimized())
    saved = json.loads(json.dumps(window.placement.to_dict()))
    window.hide()
    window.close()

    # Placement overrides the supplied geometry and mode.
    reopened = spy.Window(width=0, height=0, mode=spy.WindowMode.minimized, placement=saved)
    try:
        _wait(reopened, lambda: reopened.is_visible() and reopened.is_maximized() == maximized)
        assert reopened.placement.to_dict() == saved
        reopened.restore()
        _wait(reopened, lambda: not reopened.is_maximized())
        assert reopened.placement.to_dict() == {**saved, "maximized": False}
    finally:
        reopened.hide()
        reopened.close()


def test_hidden_placement(desktop: None) -> None:
    saved = spy.WindowPlacement(_position(), [480, 320], maximized=True).to_dict()
    window = spy.Window(placement=saved, visible=False)
    try:
        expected = {**saved, "maximized": False}
        assert not window.is_visible()
        assert window.placement.to_dict() == expected
        window.show()
        _wait(window, lambda: window.is_visible())
        assert not window.is_maximized()
        assert window.placement.to_dict() == expected
    finally:
        window.hide()
        window.close()


@pytest.mark.parametrize("maximized", [False, True])
def test_placement_json_round_trip(maximized: bool) -> None:
    placement = spy.WindowPlacement([-400, 120], [800, 600], maximized)
    state = json.loads(json.dumps(placement.to_dict()))
    assert state == {
        "normal_position": [-400, 120],
        "normal_size": [800, 600],
        "maximized": maximized,
    }
    restored = spy.WindowPlacement(state)
    assert restored.to_dict() == state
    state["normal_size"][0] = 100
    assert restored.normal_size == spy.uint2(800, 600)


def test_placement_dictionary_defaults() -> None:
    assert spy.WindowPlacement({"normal_size": [640, 480]}).to_dict() == {
        "normal_position": [0, 0],
        "normal_size": [640, 480],
        "maximized": False,
    }


@pytest.mark.parametrize("size", [(0, 300), (400, 0), (2**31, 300)])
def test_invalid_dimensions(window: spy.Window, size: tuple[int, int]) -> None:
    with pytest.raises(RuntimeError, match="Window dimensions"):
        spy.Window(width=size[0], height=size[1], visible=False)
    with pytest.raises(RuntimeError, match="Window dimensions"):
        window.resize(*size)
    with pytest.raises(ValueError, match="Window dimensions"):
        spy.WindowPlacement(normal_position=[0, 0], normal_size=size)
    with pytest.raises(ValueError, match="Window dimensions"):
        spy.WindowPlacement({"normal_size": list(size)})
    assert window.size == spy.uint2(400, 300)


def test_invalid_placement(window: spy.Window) -> None:
    with pytest.raises(ValueError, match="Window dimensions"):
        spy.WindowPlacement({})
    with pytest.raises(RuntimeError, match="Unknown key"):
        spy.WindowPlacement({"normal_size": [800, 600], "unexpected": True})
    with pytest.raises(RuntimeError, match="Cannot combine window placement with fullscreen mode"):
        spy.Window(placement=window.placement, mode=spy.WindowMode.fullscreen, visible=False)


if __name__ == "__main__":
    if "--probe-monitors" in sys.argv:
        _probe_monitor_lifetime()
    else:
        raise SystemExit(pytest.main([__file__, "-v"]))
