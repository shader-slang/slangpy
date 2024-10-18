# from kernelfunctions.backend.slangpynativeemulation import *
import os

if not "SLANGPY_DISABLE_NATIVE" in os.environ:
    from sgl.slangpy import *
else:
    from kernelfunctions.backend.slangpynativeemulation import *
