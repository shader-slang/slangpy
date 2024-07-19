# Thought process for python classes

## TLDR

Long story short, the process I suggest for integrating type methods as with copper is:

```
#Define the class with all the usual properties + explicitly declare the function
class MaterialDataGenerator(kf.Struct):
    generateMaterialSample: kf.Function
    ...

    def __init__(self, module, scene, sample_generator):
        super().__init__(module)
        ...

#Can now call the function, and everything will be passed in with fields serialised as you'd expect
module = device.loadModule("xx.slang")
md = MaterialDataGenerator(module, ...)
md.generateMaterialSample(...)
```

Potentially with the option to use decorators if inheritance isn't feasible. Lots of reasoning below that generally revolves around clarity and ability for code analysis / completion to work correctly.


## Basic class marshalling

The simplest initial support for Python classes is to ensure that they can be serialized in the same way a dictionary can. The lack of this is frustrating in various Python worlds (eg json serialization), so it'd be nice to solve for structured buffer / uniform / paramters. Recalling this example:

```
xx.slang
[Differentiable]
struct Ray {
    float3 pos;
    float3 dir;
}

[Differentiable]
float3 raytrace(in Ray ray) {
    //...hard stuff to work out the color here!...
    return color;
}
```

```
#if this can be done
color = raytrace.call({
    'pos': sgl.float3(10,20,30),
    'dir': sgl.float3(0,1,0)
})

#then this should also work
class RayTraceInput:
    __init__(self)
        self.pos = sgl.float3(10,20,30)
        self.dir = sgl.float3(0,1,0)
rti = RayTraceInput()
color raytrace.call(rti)
```

This certainly has some value to it - for example having a class that represents the scene, or a reasonable set of materials, and passing that to a shader function would be convenient. However for the same reason the `Ray` example passing a ray as dictionary has limitted use (creating an array of 1 million rays in python would be crazy), it falls down when using classes.

## Batched / tensor style calls

A more useful approach would be to mimic the tensor passing concept for dictionaries, using fields of a class instead:

```
#Assuming the colors are are made an output, then if this can be done
positions = Tensor([1,1,1], [2,2,2], ... )
directions = Tensor([0,0,1], [0,1,0], ... )
colors = Tensor()
raytrace.call(rays={'pos': positions, 'dir': directions}, colors=colors)

#then this should also work
class RayTraceInput:
    __init__(self)
        self.pos = Tensor([1,1,1], [2,2,2], ... )
        self.dir = Tensor([0,0,1], [0,1,0], ... )

rti = RayTraceInput()
colors = Tensor()
raytrace.call(rays=rti, colors=colors)
```

And due to the flexibility of Python, we could write the more elegant:

```
class RayTraceInput:
    __init__(self)
        self.pos: sgl.float3 | Tensor = None
        self.dir: sgl.float3 | Tensor = None

rti = RayTraceInput()
rti.pos = sgl.float3(1,1,1)
rti.dir = Tensor([0,0,1], [0,1,0], ... )
colors = Tensor()
raytrace.call(rays=rti, colors=colors)
```

If class marshalling is written correctly (to operate the same as dictionary marshalling), all this should simply be supported naturally. And just as with other scenarios, provided the class fields were differentiable types, the backwards call would work and result in .grad properties being added where requested.


Using snippets from Benedikts material experiment, you could end up with this in a real world scenario

Slang material generator:

```
xx.slang
struct MaterialDataGenerator {
    ...
}

void generateMaterialSample(
        in MaterialDataGenerator generator,
        int index,
        <...
    )
{
    ...
}
```

Python class that matches it, and call to the generateMaterialSample function

```
class MaterialDataGenerator:
    def __init__(self, scene, sample_generator):
        self.gridDim = [256, 256, 1, 1]
        self.uvRange = [0.0, 0.0, 1.0, 1.0]
        self.useNormalMapping = True
        self.sampleTexelCenter = False
        self.sampleBufferContent = False
        self.skipEvaluation = False
        self.sampleCount = 1
        self.materialID = 1
        self.batchIndex = 0
        self.colorAugmentationRatio = 0.0
        self.balancedSamplingRatio = 1.0
        self.materialSamplingStrategy = int(falcor.NeuralMaterialSamplingStrategy.UniformRusinkiewicz)
        self.sourceResolution = 2048
        self.minMaxMipLevel = [0, 0]
        self.angularPerturbationWidth = 0.0
        self.numPerturbationSamples = [1, 1]
        self.scene = scene
        self.sample_generator = sample_generator

    def set_shader_globals(self, rootvar):
        self.scene.bind_shader_data(rootvar["gScene"])
        self.sample_generator.bind_shader_data(rootvar)

mdg = MaterialDataGenerator(scene, sample_generator)
indices = Tensor(<list of indices>)

generateMaterialSample = kf.get_function(mymodule, "generateMaterialSample")
generateMaterialSample.call(mdg, indices)
```

Ultimately, provided class marshalling still works, this should fall naturally out of the kernel functions api design.

## Type methods

### Explicit type method binding

`copper` also supports directly calling methods on types. This should be relatively trivial to implement given the above code works - the big question is how to cleanly implement it api side. The most basic approach would be to make it very explicit:

The generate function is moved into the slang structure (as is the case in the real code base)

```
xx.slang
struct MaterialDataGenerator {
    ...

    void generateMaterialSample(
        in MaterialDataGenerator generator,
        int index,
        <...
    ) { ... }
}
```

The python side explicitly wraps the generateMaterialSample function, just as with global functions, and it can now be
called as an instance:

```
class MaterialDataGenerator:
    def __init__(self, kf_module, scene, sample_generator):
        self.generateMaterialSample = kf_program.get_function(self, "MaterialDataGenerator.generateMaterialSample")

    def set_shader_globals(self, rootvar):
        ...

kf_module = kf.Module(mymodule)
mdg = MaterialDataGenerator(kf_module, scene, sample_generator)
indices = Tensor(<list of indices>)
mdg.generateMaterialSample.call(indices)
```

(this may require some Python tweaking, but would function in principle)

This has pros and cons:
- Con: oppurtunity for bug forgetting to pass 'self' to kf.get_function
- Con: need to repeat several names multiple times
- Con: the need to add '.call' stops it looking like a simple class function
- Con: had to introduce and pass a new 'kf Program' type
- Pro: very explicit, clear what's happening
- Pro: explicit creation of the property and kf call means code completion / type checking will be seamless
- Pro: enforcing the '.call' is consistent with other api, and allows insertion of options / state setting etc
- Pro: creating object explicitly allows early setup of unchanging options / settings on the function object

### Reduce boiler plate through inheritance / decorators

Using inheritance and/or decorators, some repetition / bug oppurtunties could be removed immediately without harming clarity:

```
class MaterialDataGenerator(kf.Struct):
    def __init__(self, module, scene, sample_generator):
        super().__init__(module)
        self.generateMaterialSample = self.get_function(module, "generateMaterialSample")

    def set_shader_globals(self, rootvar):
        ...

module = device.loadModule(...)
mdg = MaterialDataGenerator(module, scene, sample_generator)
indices = Tensor(<list of indices>)
mdg.generateMaterialSample.call(indices)
```

Here:
- Requiring an inherited type means adds a new well defined `get_function`, allowing code completion / type checking to work
- The matched type is naturally inferred from the class name, though could be overriden in various ways
- The call to `super` gives clear reason for `module` to be passed in and the typing involved

Using a class decorator instead of inheritance could achive exactly the same goal and not interfere with other inheritance chains. Despite the reduction in code, I'm less a fan of this, as it introduces functions at run time that IDEs, linters, code completion and type checking don't understand. However it's functionally identical and could even be provided as an alternative in the event inheritance was impractical.

Ultimately, using at least one of these methods is a no brainer, and they don't actually prevent the use of the earlier more explicit approach if needed for some edge case.

### Auto detecting functions

Given reflection allows us to list the functions on a Slang struct, arguably the explicit declaring of functions is unnecesary. i.e. the class could look like this, and have the 'generateMaterialSample' function automatically detected:

```
class MaterialDataGenerator(kf.Struct):
    def __init__(self, module, scene, sample_generator):
        super().__init__(module)
```

This does make me nervous however - it means we're doing really serious amounts of 'secret' work behind the scenes! A middle ground though might be to just require the declaration of functions so they get hooked up automatically. A middle ground, which is a little dirty but would probably trick code analysers into doing the right thing would be to require functions be defined as class variables:

```
class MaterialDataGenerator(kf.Struct):
    generateMaterialSample: kf.Function

    def __init__(self, module, scene, sample_generator):
        super().__init__(module)
```

Internally, on construction each class function would be looked up and a corresponding instance version would be created. This would likely allow code analysers to accept the following:

```
mdg = MaterialDataGenerate(...)
mdg.generateMaterialSample.call(...)
```

My sense is is that this level of explicit declaration is the reasonable one. As with global functions, it requires a user only to write the name of the function to bind to only once, whilst maintaining a clarity in the code base that other programmers and code analysers can easily parse.

If desired, we could even allow a user to explicitly define what parameters they expected the function to take. This could be used as an optional mechanism to improve binding checking and help with code completion.

### Call operators

With both global and class methods, thus far all calls to a kernel function have been of the form `x.call(args)`. In theory, there is no reason why the `.call` bit could be removed, and the `__call__` operator overriden instead. in the above example, this would take it down to:

```
mdg = MaterialDataGenerate(...)
mdg.generateMaterialSample(...)
```

I personally am not keen on this for not entirely justifiable reasons! I guess it's because it is hiding something big behind something tiny, and isn't really consistent with the other method chaining aspects of the API. But, it is standard practice in python ML with modules and functions, so maybe I'm just being picky!








