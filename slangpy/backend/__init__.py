# pyright: reportWildcardImportFromLibrary=false, reportUnusedImport=false
import sys
import os

try:
    from sgl import *
    BACKEND = "SGL"
except ImportError:
    from .falcorwrapper import *
    BACKEND = "Falcor"

current_module = sys.modules[__name__]
