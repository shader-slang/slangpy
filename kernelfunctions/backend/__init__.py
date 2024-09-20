# pyright: reportWildcardImportFromLibrary=false

try:
    from sgl import *
    BACKEND = "SGL"
except ImportError:
    from .falcorwrapper import *
    BACKEND = "Falcor"
