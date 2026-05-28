# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import slangpy as spy


def test_keycode_world_1_and_2_exist():
    # GLFW defines these in the "printable" range:
    #   GLFW_KEY_WORLD_1 = 161 (non-US #1)
    #   GLFW_KEY_WORLD_2 = 162 (non-US #2)
    #
    # When reported via KeyboardEvent.key, the Python wrapper must be able to
    # materialize these as valid KeyCode enum values (no ValueError).
    assert spy.KeyCode(161) == spy.KeyCode.world_1
    assert spy.KeyCode(162) == spy.KeyCode.world_2
