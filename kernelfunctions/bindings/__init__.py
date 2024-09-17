import runpy

# Call all binding modules to register types
runpy.run_module("kernelfunctions.bindings.buffertype")
runpy.run_module("kernelfunctions.bindings.diffpairtype")
runpy.run_module("kernelfunctions.bindings.structtype")
runpy.run_module("kernelfunctions.bindings.structuredbuffertype")
runpy.run_module("kernelfunctions.bindings.texturetype")
runpy.run_module("kernelfunctions.bindings.valuereftype")
runpy.run_module("kernelfunctions.bindings.valuetype")
runpy.run_module("kernelfunctions.bindings.arraytype")
