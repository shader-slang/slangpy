# pyright: reportWildcardImportFromLibrary=false, reportUnusedImport=false
import kernelfunctions.backend.slangpynativeemulation as slangpynative
import sys
import os

try:
    from sgl import *
    BACKEND = "SGL"
except ImportError:
    from .falcorwrapper import *
    BACKEND = "Falcor"

current_module = sys.modules[__name__]

# if not "SLANGPY_DISABLE_NATIVE" in os.environ and hasattr(current_module, "slangpy"):
#    slangpynative = getattr(current_module, 'slangpy')
# else:
#    import kernelfunctions.backend.slangpynativeemulation as slangpynative

# For now using the internal native emulation
