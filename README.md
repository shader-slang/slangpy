# kernelfunctions



# Overview

Currently we have several implementations of systems that integrate python with slang, and various projects sitting on top of them:
- **Slang Torch**: CUDA based python project for compiling and calling CUDA functions written in slang and using them with pytorch
- **Falcor**: Does absolutely everything plus a bit more
- **SGL**: Low level python first graphics api
- **SLGU**: Thin wrapper around SGL with basic concept of structured buffers that have associated gradients and modules designed for differential calls
- **Copper**: Benedict's new experiments in creating a simpler mechanism to call Slang functions from Python, sitting on top of Falcor

Outside of these, we've got projects that can be divided up in a few different ways - some use Falcor, some use SGL, some just use SlangTorch with PyTorch. These divide up further with various approaches to forward/backward processes and PyTorch vs custom training mechanisms. Looking at the Slangtorch examples (includes learned mip maps), neural textures, neural materials, difftrace, gaussian splats and falcor ray tracers gives a wide range of examples of problems solved, and how code has 'ended up' being written under certain constraints.

Having reviewed the various options, copper represents the best high level approach in creating a clean and usable interface from Python to Slang via SGL. I've called this kernel functions for now, as its too boring a name to possibly use, so it won't accidently end up being stuck with!

See Benedikt's write up for more info on `copper`: https://gitlab-master.nvidia.com/bbitterli/Falcor/-/tree/copper2/scripts/internal/NeuralAppearance/python/copper2?ref_type=heads

# Calling kernels and batching

Derived from `copper`, Benedikts observation, which we all generally agree with, is that the boiler plate involved in writing explicit compute kernels needs to go. A user should be able to write this function:

```
// xx.slang
float add(float x, float y)
{
    return x+y
}
```

And call it from Python, whilst being allowed to pass in either a constant or an array-like structure for either x or y, and automatically be returned either a single float or an array-like structure accordingly. i.e.

```
# yy.py

#pass in 2 constants, get returned a constant
x = 10
y = 20
z = add(x,y)
#z -> 30

#pass in 1 list and 1 constant, get returned list
x = [10,20,30]
y = 20
z = add(x,y)
#z -> [20,30,40]

#pass in 2 list
x = [10,20,30]
y = [40,50,60]
z = add(x,y)
#z -> [50,70,90]

#pass in 2 mismatching lists, get an error
x = [10,20,30]
y = [40,50,60,70]
z = add(x,y)
#z -> ERROR
```

Of course, in GPU/Python world, rather than lists we'd be likely dealing with structured buffers, tensors or numpy arrays.

## in / out / inout Parameters

Slang supports the full range of in, out and inout parameters. These would be supported in the API exactly as how one would expect them to operate:
- in: works as you'd expect
- out: writes back result (as you'd expect) - would fail to write to broadcast value
- inout: again, as you'd expect

We have discussed whether function return values should actually be forbidden, with results _only_ returnable via an `out` parameter. This has considerable benefits both in terms of matching the PyTorch auto diff style, and reducing ambiguities in how values should be returned in more complex batching situations. That said, in many cases this is an unnecesary restriction and would be nice to avoid.

## More complex batching

When we start thinking in terms of batch tensor operations such as those involved in training, it's also reasonable to start thinking about multi-dimensional versions of the same batching:

```
yy.py

#pass in a multi dimensional tensor, such as that used for training 2 entries in a batch
#of 1D data
x = [ [10,20,30], [40,50,60] ]
y = 20
z = add(x,y)
#z -> [ [30,40,50], [60, 70, 80] ]

#which leads to lower dimensional uniforms
x = [ [10,20,30], [40,50,60] ]
y = [1,2,3]
z = add(x,y)
#z -> [ [31,42,53], [41, 42, 63] ]
```

Thus far we have dealt explicitly with a function that takes none-array like structures, but as detailed by Benedikt, it gets complex when multidimensional structures turn up:

```
xx.slang
float2 sample(int position, float2[] source)
{
    return source[position]
}

```

In the above example, the source array could just as easily have been a structured buffer, tensor or even a texture. I am unclear yet as to whether there are actual ambiguities that can be created, or whether for any given set of inputs you can always identify either a single correct answer, or a conflicting batch size.

# Core API structure

Whilst the above examples involve single function calls to methods, and passing in pure python lists as batched parameters is elegant, it does remove oppurtunities to customize / defer calls. It can also create ambiguities in how certain parameters are passed to Slang. With that in mind, we will implement the Pythonic method-chaining pattern for function calls:

```

#use SGL to load a SlangModule object
my_module = device.loadModule("xx.slang")

#extract a 'kernel function' from the 'add' function defined in xx.slang
add_func = kf.get_function(my_module, "add")

#the kernel function supports method chaining to configure a call before triggering it
add_func
    .set({})                #set some properties before calling
    .options({})            #possible way of configuring specific properties of the call
    .typeconformance({})    #may allow specification of type conformances as part of call
    .call(args)             #all real work is deferred until the actual call
```

As each method returns none-mutating state, it also allows users to preconfigure their calls

```
#configure once
configured_add_func = add_func
    .set({})            
    .options({})        
    .typeconformance({})

#call multiple times
configured_add_func.call(args1)
configured_add_func.call(args2)

#also support shothand through the use of Python `__call__` operator
configured_add_func(args1)
configured_add_func(args2)

```

This offers up oppurtunities to trigger the call in other ways, such as adding it to an SGL command list:

```
command_buffer = device.create_command_buffer()
with command_buffer.encode_compute_commands() as encoder:

    #api could require explicit version of call to add to command list
    configured_add_func.append_to(encoder, args)

    #or we could make it implicit supplying an encoder argument, though
    #I personally think this is bug city!
    configured_add_func.call(encoder, args)

command_buffer.submit()
```

As shown in the examples, the API will also override the Python `__call__` operator, allowings users to treat the kernel function just like any other function where it makes sense.

## Backwards Pass

Mimicking how existing machine libaries operate, the differential can be supported with a backward property, which is simply another callable function:

```
add_func.backward
    .set({})
    .options({})
    .call()
```

And, probably quite helpfully, the backwards pass for a preconfigured function could inherit its properties from the primal pass:

```
#calculate primal result
result = configred_add_func.call(a,b)

#call the backward function to attach grads (see differentiable types below)
configured_add_func.backward.call(a,b,result)

```

Were the slang function none-differentiable, the .backward function would simply be `None`.

Conveniently, as we're simply wrapping Slang function calls, if a custom differential were provided for the function in question, it would naturally be called as part of the backward pass above.

Thus far I am inclined to suggest we leave forward differentials out of the Python api until required - they can of course still be calculated within a slang program, however I'm concerned that by attempting to wrap a Pythonic api around both forward and backward differentials we'd end up creating confusion unnecesarily, especially given 'forward' typically refers to the feed-forward step in the context of machine learning frameworks.

## Explicit batched types

Whilst the earlier ability to pass pure lists in as batched parameters is convenient, it has the potential to cause confusion when, for example, calling a function that takes an array or even just multi dimensional types such as vectors. 

To handle this, we _could_ implement a model similar to PyTorch's `is_tensor_like`, and add a `kf.is_batchable_like`, along with a default `Batchable` type. Thus to support batching on a list, at a minimum you would be required to wrap it in a Batchable:

```
a = 10
b = Batchable([1,2,3])
result = add(a,b)
```

As `kf.is_batchable_like` would return true for any tensor-like type, this would also be valid (and probably more common):

```
a = 10
b = torch.Tensor([1,2,3])
result = add(a,b)
```

I am currently undecided on this point - it feels clean, but raises the question of what types should be 'batchable'. A numpy array? A structured buffer? If so, why these types but not a list?

## Differentiable types

Most machine learning libraries make the observation that a parameter and its associated partial derivate  are deeply linked, and thus attach 1 to the other. For example, in PyTorch a `Tensor` object has a `.grad` property, which is also a `Tensor`. Slang makes a similar observation in its implementation of a `DifferentialPair`. 

It makes sense again to provide `kf.is_differentiable_like`, which returns true if the associated type supports having `.grad` and `.needs_grad` properties. Again this could be true by default for a PyTorch `Tensor`, however would also allow us to create wrappers for scalars or structured buffers. For example, a possible implementation could involve

```
a = kf.differentiable(5) #returns the correct container for a differntiable scalar
b = kf.differentiable(my_structured_buffer) #returns a wrapper around a structured buffer with grad
c = torch.Tensor([1,2,3]) #just works!
```

## Remembering the basics!

Before getting all clever, it's important to remember that we want to make the simpler cases easy as well - partly as that's just useful, but also when the fancy functionality of a library doesn't quite cut it, users need to go back to writing simpler more bespoke code themselves. For example:

```
xx.slang
void fill(int2 pixel, float3 color, RWTexture2D<float3> texture) {
    texture[pixel] = color
}
```

Should provide a mechanism for setting `pixel` to a range of values. This could utilize the standard `shape` concept from `numpy`:

```
fill.call(pixel=Shape(256,256), color=float3(1,0,0), texture=mytex)
```

Even a prewritten compute kernel such as the following should be able to take advantage of convenient buffering and pythonic calling conventions:

```
uniform RWTexture2D<float3> out;
uniform float3 color;
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    uint2 pixel = tid.xy;
    uint2 dim;
    out.GetDimensions(dim.x, dim.y);
    if (any(pixel >= dim))
        return;
    out[pixel] = color;
}
```

```
main.set(color=float3(1,0,0), texture=mytex)
    .threads(mytex.width,mytex.height)
    .call()
```

# Simple path tracer example

This demonstrates the process of converting a simple, none differentiable path tracer that has already been written without using kernel functions. It'd probably come to us looking like the following:

```
ParameterBlock<Scene> g_scene;
RWTexture2D<float4> g_output;
uniform uint g_frame;


[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    uint2 pixel = tid.xy;
    uint2 dim;
    g_output.GetDimensions(dim.x, dim.y);
    if (any(pixel >= dim))
        return;

    const uint spp = 4;
    float3 L = float3(0);
    for (uint i = 0; i < spp; i++) {
        RNG rng = RNG(pixel, g_frame * spp + i);
        float2 uv = float2(pixel + rng.next_2d()) / float2(dim);
        uv.y = 1 - uv.y;
        Ray ray = g_scene.camera.get_ray(uv);
        Path path = Path(pixel, ray, rng);
        trace_path(path);
        L += path.L;
    }
    L /= spp;
    g_output[pixel] = float4(L, 1);
}
```

Ideally this could be made to work trivially, just by 'calling' the main function, but then enhanced if need be. For example, our first attempt might be:

```
yy.py
main.set({
        'g_scene' = bla,
        'g_output' = MyTexture,
        'g_frame' = 10
    })
    .threads(MyTexture.width,MyTexture.height)
    .call()
```

However if re-written with 'kernel functions' in mind, we might pair back to to:

```
xx.slang
ParameterBlock<Scene> g_scene;

void raytrace(frame: uint, pixel: uint2, out float4 res)
{
    const uint spp = 4;
    float3 L = float3(0);
    for (uint i = 0; i < spp; i++) {
        RNG rng = RNG(pixel, g_frame * spp + i);
        float2 uv = float2(pixel + rng.next_2d()) / float2(dim);
        uv.y = 1 - uv.y;
        Ray ray = g_scene.camera.get_ray(uv);
        Path path = Path(pixel, ray, rng);
        trace_path(path);
        L += path.L;
    }
    L /= spp;
    res = float4(L, 1);
}

yy.py
raytrace
    .set(g_scene=bla)
    .call(
        frame=10,
        pixel=MyTexture.shape,
        res=MyTexture
    ) 
```

In this case though a key optimization has been lost - the original ray tracer quite deliberately used cache coherent blocks of 16x16 pixels. This suggests still we'd want the ability to add additional options before the call to control code gen:

```
yy.py
raytrace
    .set(g_scene=bla)
    .options(numthreads=[16,16])
    .call(
        frame=10,
        pixel=MyTexture.shape,
        res=MyTexture
    ) 
```

A final note there is the significant parameter block containing the whole scene. In theory it could be passed as an argument, however a user may not wish to structure code in this highly functional way, and an API shouldn't force users to a certain way of thinking unless absolutely necessary.

# Complex Types / Type Marshalling

Thus far we've focussed on simple types like 'float', however we need to support the full range of types that Slang supports. SGL already contains extensive type marshalling support (though needs some additions), so is well suited to the problem. For this section, we can assume that the basic process of converting a python dictionary to/from a slang structure is solved (as it pretty much is in SGL). That is, it is currently possible in SGL to call:

```
    computekernel.dispatch(
        thread_count=[N, 1, 1],
        vars={"processor": {"a": buffer_a, "b": buffer_b, "c": buffer_c}},
    )
```

And it will 'just work'. Whilst this process isn't entirely flushed out in SGL, it needs to at some point, and and is not too complex (albeit a little tedious!) to get going!

## Simple structures / batching

A basic call in a differentiable ray tracer might look like the following:

```
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

SGL already has a 'float3' type, so in the simplest, none-batched call to this function, we'd probably end up with:

```
#fire a ray along the z axis from [10,10,0]
color = raytrace.call({
    pos = sgl.float3(10,10,0),
    dir = sgl.float3(0,0,1)
})
```

Internally this would either need to marshal the ray into a structured buffer of some form, or write it directly as a uniform. 

Batching could operate as expected if providing a structured buffer or batchable list of rays:

```
rays = kf.Batchable(<some list of ray classes/dictionaries>)
color = raytrace.call(rays)
#color -> list of colors
```

It's worth noting here that there is an ambiguity in what form the color array should be returned - a tensor? a buffer? a list? It's in this case that enforcing `out` parameters rather than return values would be effective:

```
xx.slang
[Differentiable]
void raytrace(in Ray ray, out float3 color) {
    //...hard stuff to work out the color here!...
}

yy.python
#As user allocates return values up front, typing is no longer ambiguous
rays = <a structured buffer of rays>
colors = <a structured buffer of float3s>
raytrace.call(rays, colors)
```

## Differentials

Whilst it would require moving to differentiable types, the backwards process then becomes fairly clear:

```
rays = kf.differentiable(<a structured buffer of rays>, needs_grad=True)
colors = <a structured buffer of float3s>
raytrace.call(rays, colors)
raytrace.backwards.call(rays, colors)

#rays.grad-> buffer of Ray.Differentials
```

While _outputs_ would by default have a gradient of 1, specification of its gradient will also be supported, to allow a kernel function to part of a larger back-propagation operation.

## Tensors / AOS / SOA

When applied to machine learning, it is unlikely that you'd have a large array of Python 'ray' objects / dictionaries. Rather, you'd represent the positions / directions in tensors. This could be done in various ways

As an array-of-structures of various layouts:
- 1*1D tensor, completely flat, [ pos.x,pos.y,pos.z,dir.x,dir.y,dir.z, repeat for more rays ]
- 1*2D tensor, flat entry per ray, [ [pos.x,pos.y,pos.z,dir.x,dir.y,dir.z], ... ]
- 1*3D tensor, 2 vectors per ray, [ [[pos.x,pos.y,pos.z],[dir.x,dir.y,dir.z]], ... ]

Or a more structure of arrays form:
- 2*2D tensor, [ [pos.x,pos.y,pos.z], ... ],  [ [dir.x,dir.y,dir.z], ... ]

Plus a million other combinations of varying degrees of flatteness and AOS vs SOA mappings!

Whilst we could attempt to get **really** clever here, my inclination is to start simple and allow:

```
positions = Tensor([1,1,1], [2,2,2], ... )
directions = Tensor([0,0,1], [0,1,0], ... )
colors = Tensor()
raytrace.call(rays={'pos': positions, 'dir': directions}, colors=colors)
```

We could take this to the extreme, and attempt to allow 'auto flattening' to the correct shape, or digging down deeper into structures so pos.x, pos.y, and pos.z could all be separate tensors. I think some of this will be necessary and fall out naturally, but we should take baby steps - it has the potential to grow into a can of worms. Certainly we should enforce dimensionality matching, even if we dive into clever typing straight off.

## Python classes

As a base line, all the rules above for marshalling dictionaries and representing fields as scalars or tensors etc should all function identically for classes. i.e. this example of nicely typed Python should function perfectly:

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

yy.py
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

Just as with dictionaries, all we're doing here is defining the fields that'll be fed to the `Ray` input for each call to `raytrace` using fields of a Python object.

## Class methods

Calling methods directly on Slang structs through the use a python method is supported by inheriting from the `kf.Struct` type, and listing the methods to bind as class fields.

A slang struct and method that use both parameters and internal state:

```
xx.slang
struct Scaler
{
    float scale;

    void scale_value(inout float val) { val *= scale; }
}
```

Python class wraps the state and defines the function it wishes to bind to:

```
class Scaler(kf.Struct):
    scale_value: kf.Function

    def __init__(self, module, scale):
        super().__init__(module)
        self.scale = scale

```

The class function is now usable just like a global function:

```
mymodule = device.loadModule('xx.slang')
scaler = Scaler(mymodule, 100.0)
numbers = Tensor([1,2,3,4])
scaler.scale_value.call(numbers)
#numbers -> Tensor([10,20,30,40])
```

Or, using the `__call__` override:

```
scaler.scale_value(numbers)
```

To keep things simple, the initial version of the API will treat class members as const, thus requiring that fields be set to writable buffers to be modified. If this turns out to be an issue, or it is simple to implement, we will expand functionality to include write back of any field.

Question: should we utilize decorators instead of / as an alternative to inheritance? This approach has pros and cons:
- Pro: `module` parameter automatically inserted + handled, so no need to remember to call `super`
- Pro: doesn't interfere with other inheritance chains
- Con: `IDE / auto complete` can not highlight / document the need for the module parameter

## Tensor type

Whilst I've referred to 'batchable lists' in this document, the reality is the vast majority of interaction with kernels would involve either:
- Broadcast scalar constants
- Graphics primitives (eg textures / buffers)
- Tensors

SGL doesn't currently have strong support for tensors, and Slang has basic support for a `TensorView`. To support  future work, we should introduce a tensor type as a core concept in both SGL and Slang with conversions to/from PyTorch tensors when necessary.

## Accumulators

Gradient accumulation during a backwards pass is the obvious case for an accumulator, however in the more general case any broadcast `out` parameter could support them. This would be particularly valuable in cases of bespoke training, where gradient calculations may not be part of an explicit 'backwards' pass.

<b>Note: This aspect of the API is 'theory' - will probably need flushing out once we've made progress on the basics!</b> 

In general any out parameter could be used for broadcasting:

```
xx.slang
void myfunc(float3 input, out float3 result)
{
    result = input * 10
}

yy.py
#by default, this should clearly fail - the inputs suggests 3 calls, but only asking for 1 output
inputs = Tensor([1,2,3],[4,5,6],[7,8,9])
result = sgl.float3()
myfunc.call(inputs, result)
# ERROR: Mismatched batching

#potential api to request the result was accumulated
myfunc.accumulate(result='sum')
      .call(inputs, result)
# result -> [10+40+70, 20+50+80, 30+60+90]
```

In a more complex real world context, training and accumulating gradients of a mini batch, we'd end up with something along the lines of:

```
batch_inputs = Tensor([ [..batch entry 0 data..], [..batch entry 1 data..], ... ], needs_grads=True)
batch_outputs = Tensor(batch_inputs.shape)

myfunc.call(input=batch_inputs, result=batch_outputs)

#populate input grads as normal
myfunc.backwards
    .call(input=batch_inputs, result=batch_outputs)
#input.grads -> gradients per mini batch entry

#populate input grads through summed accumulation
myfunc.backwards
    .accumulate({'input.grad': 'sum'})
    .call(input=batch_inputs, result=batch_outputs)
#input.grads -> accumulated input gradients across the whole mini batch
```


## Pytorch integration

The principle requirement for PyTorch integration is that the kernel function needs to interact seamlessly with PyTorch autograd. When it comes down to it, this involves 2 engineering problems:
- Ensuring tensors are tracked by auto grad by calling kernel functions via an `autograd.Function.apply` call
- Supporting the idea of a context in which an operation is occuring, so a user doesn't have to remember which tensors were involved in a forward pass when calling a backwards one

I am reluctant to enforce context tracking at the base layer of the API - not all use cases of kernel functions will want this, and it would be intrusive. As such, I propose we support creation of a PyTorch wrapper:

```
#Convert kernel function to PyTorch one by calling torch() instead of call()
Add = add_func.torch()

#Due to none-mutable state storage, would also support configuring
Add = add_func
    .set({})                #set some properties before calling
    .options({})            #possible way of configuring specific properties of the call
    .typeconformance({})    #may allow specification of type conformances as part of call
    .torch()

#Can now call Add, just as with any normal custom auto grad function
a = Tensor(bla)
b = 10
result = Add(a, b)
result.backward()

#a.grad- -> dout / da
```

Internally this would be relatively simple to implement. The call to `torch()` would generate a function that could gain access to the input/output tensors. These would then be passed to a call an `autograd.Function.apply` call in the correct order. It's made doubly easy, because Benedikt already solved most of this in `copper` :)

Note: if pytorch module was not found, this would simply throw an exception, thus allowing kernel functions to operate correctly in a project without pytorch installed.

## None-pytorch projects / Custom optimizers

Due to the huge performance gains it's reasonable to assume the trend of custom optimizers will continue. Whilst it's not the job of kernel functions to _solve_ this, they should still represent an increase in usability over pure calls to compute.

<b>NOTE: This is my current reading of the Gaussian code - may not be correct :) </b>

The Gaussian project is a good example of this, which uses (Marco's?) lightweight `sglu` library as a base interface to SGL. At its core, `sglu` implements:
- A structured buffer wrapper that can have associated grads
- A hierarchical `Module` base class that can bind parameters and (critically) return a flat array of all structured buffers in itself and any children that require gradients

Similar to the PyTorch Adam implementation, the Gaussian version takes a list of parameters that need training. Within the training loop, primals and gradients are calculated as usual and the optimizer does its job. The critical difference between this setup and PyTorch is the lack of autograd - it is down to the modules to generate the correct gradients and write them to the correct buffers.

Provided we ensure minimal overheads, out of the box kernel functions would improve a few areas without causing any hindrance:
- Boiler plate for slang kernels and associated loading/calling code would be reduced. 
- The custom structured buffer type would not be needed
- Where desired, more flexible tensor structures could be used, reducing the need for manual indexing
- Up front knowledge of buffer param size counts wouldn't be necessary for gradient allocation (would be implicit based on calls)

Aspects we'd need to ensure still worked / were fast are:
- To support the custom gradient generation, it would need to be easy to access and pass gradient buffers to kernel functions
- Setting constants up front is useful, but we need to ensure modifying globals every frame has no new overheads
- Robust control over thread grouping for dispatch

In effect, a version 1 would probably take strong advantage of the simpler calling mechanisms, but gain less from the simplified calls to backward propagation. 

Going forward, natural extensions would be:
- Implement some form of call graph around kernel functions to replace the `Module` concept
- Accumulator support to remove some of the boilerplate around gradient accumulation
- Look at providing 'fused' forward+backward kernel support, as used by the differential rasterizer

## Raytracing Pipeline

The ray tracing API would be relatively trivial to fit to the same model. At the very least, this shader could be built:

```
raytrace.slang

void missed(inout Payload payload)
{
    payload.color = float3(0, 0, 0);
}

void hit(inout Payload payload, BuiltInTriangleIntersectionAttributes attribs)
{
    payload.color = float3(attribs.barycentrics, 0);
}

void ray_gen(uint2 pixel, out float4 color)
{
    uint2 dim = DispatchRaysDimensions().xy;
    float2 uv = float2(pixel) / float2(dim);

    RayDesc ray;
    ray.Origin = float3(uv * 2 - 1, 1);
    ray.Direction = float3(0, 0, -1);
    ray.TMin = 0;
    ray.TMax = 2;

    Payload payload = {};
    TraceRay(tlas,0,0xff,0,0,0,ray,payload);
    color = float4(payload.color, 1.f);
}
```

With the following simple API to build a 'ray gen' kernel function, and wrap corresponding callbacks if need be

```
raytrace.py

#make a ray gen kernel function
ray_gen = kf.get_ray_gen(module, "ray_gen",  miss="missed", closesthit="hit")

#<do all the tedious TLAS setup that we have to make better>

ray_gen
    .set("tlas": my_tlas)
    .options(
        hit_groups = [bla],
        max_recursion = 1,
        max_payload_size = 12    
    )
    .call(pixel=MyTexture.Shape, color=MyTexture)
```

With work, common patterns such as converting pixel to uv could probably also be wrapped up cleanly.

## Graphics Pipeline

This'll be left as a TODO for a little while, but I envisage it being wrapped up in the same style. Ultimately render state is just a set of options, and vertex / index buffers are simply parameters to a function call. There's no reason it can't be mimic a similar pattern.

## Render Graphs / Fusing

Whether it belongs in a kernel-functions package or a level higher, render graphs would be well supported by this model. As with PyTorch, we would introduce a `.node` method to convert a kernel function to a render graph node. At this point, standard Pythonic call tracing techniques could be used to build simple graphs:

```
MyGraph = kf.Graph(<minimal input/output information>)

# Create nodes from 3 functions
NodeA = function_a.node()
NodeB = function_b.node()
NodeC = function_c.node()

#Read inputs from graph
(input1, input2, input3) = MyGraph.get_inputs()

#Call Node A (returns 2 outputs) and Node B
(output1, output2) = NodeA(input1)
output3 = NodeB(input2)

#Feed all 3 outputs into Node C
MyGraph.set_outputs(NodeC(output1, output2, output3))

# Can now call the graph, and (with help from Sai working out chain rule 
# for render graphs), a corresponding backwards:
result = MyGraph.call(args)
MyGraph.backwards.call(args, result)

# And because we're still in kernel function land, we could convert the graph to PyTorch
MyPyTorchGraph = MyGraph.torch()

# Or, it could be turned into a node for another graph, of which we could have a library!
MyReusableNode = MyGraph.node()
```

As an aside, the strongly data/reflection driven nature of kernel functions also leaves us open to render graph visualizers/debuggers or even graphical user interfaces for building them.

Fusing is simply the process of taking a graph like the one above, and rather than calling it as a sequence of dispatches with buffer transfers, combining calls into a single kernel and triggering them with a single dispatch. Whether this process was automatic or not would probably just depend on reliability / compilation costs, but it would be easy to implement at a later date if the graph API was clearly defined.

## Dynamic state

Thus far the API has provided a robust means to set globals before a call:

```
myfunc.set({...})
      .call(args)
```

However most real time situations have a lot of dynamic state. Remembering to update / pass this state consistently is a source of bugs.

It's conceivable we could add a simple mechanism to 'hook' a call:

```
def callback(context):
    ...use context to bind / configure dynamic state...

my_configured_func = myfunc
    .set({... unchanging globbal state ...})
    .hook(callback)
    .call(args)

#will now trigger 'callback' every time its called
my_configured_func.call(args)
```

In this case, we could equally allow the hook mechanism to bind to a class, perhaps with an explicit function name:

```
class Scene:

    def render_callback(context):
        ...

scene = Scene()

my_configured_func = myfunc
    .set({... unchanging globbal state ...})
    .hook(scene, Scene.render_callback) # probably there's a better way!
    .call(args)

#will now trigger 'render_callback' on the scene instance every time its called
my_configured_func.call(args)
```

There is a question over whether this functionality is too high level for kernel functions, however the oppurtunity to hook into certain calls is likely to be useful to any higher level APIs.

