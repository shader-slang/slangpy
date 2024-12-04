import runpy

# Call all binding modules to register types
runpy.run_module("slangpy.builtin.valuetype")
runpy.run_module("slangpy.builtin.valuereftype")
runpy.run_module("slangpy.builtin.diffpairtype")
runpy.run_module("slangpy.builtin.buffertype")
runpy.run_module("slangpy.builtin.structtype")
runpy.run_module("slangpy.builtin.interfacetype")
runpy.run_module("slangpy.builtin.structuredbuffertype")
runpy.run_module("slangpy.builtin.texturetype")
runpy.run_module("slangpy.builtin.arraytype")
runpy.run_module("slangpy.builtin.resourceviewtype")
runpy.run_module("slangpy.builtin.accelerationstructuretype")
runpy.run_module("slangpy.builtin.rangetype")
