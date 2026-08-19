# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys


if os.environ.get("SLANGPY_TEST_DISABLE_TORCH"):
    # Make installed Torch packages behave as if they were unavailable. This
    # runs before the SlangPy pytest plugin imports the native extension, so the
    # bridge and individual tests consistently observe the same environment.
    for package in ("torch", "slangpy_torch"):
        if sys.modules.get(package) is not None:
            raise RuntimeError(f"{package} was imported before it could be disabled")
        sys.modules[package] = None


pytest_plugins = "slangpy.testing.plugin"
