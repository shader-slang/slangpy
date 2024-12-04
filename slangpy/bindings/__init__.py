import runpy

# Call all binding modules to register types
runpy.run_module("slangpy.bindings.valuetype")
runpy.run_module("slangpy.bindings.valuereftype")
runpy.run_module("slangpy.bindings.diffpairtype")
runpy.run_module("slangpy.bindings.buffertype")
runpy.run_module("slangpy.bindings.structtype")
runpy.run_module("slangpy.bindings.interfacetype")
runpy.run_module("slangpy.bindings.structuredbuffertype")
runpy.run_module("slangpy.bindings.texturetype")
runpy.run_module("slangpy.bindings.arraytype")
runpy.run_module("slangpy.bindings.resourceviewtype")
runpy.run_module("slangpy.bindings.accelerationstructuretype")
runpy.run_module("slangpy.bindings.rangetype")
