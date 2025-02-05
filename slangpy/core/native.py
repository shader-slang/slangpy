# SPDX-License-Identifier: Apache-2.0
# from slangpy.backend.slangpynativeemulation import *

import os

if not "SLANGPY_DISABLE_NATIVE" in os.environ:
    from sgl.slangpy import *
else:
    from slangpy.backend.slangpynativeemulation import *
