# kernelfunctions



# Overview

Currently we have several implementations of systems that integrate python with slang, and various projects sitting on top of them:
- **Slang Torch**: CUDA based python project for compiling and calling CUDA functions written in slang and using them with pytorch
- **Falcor**: Does absolutely everything plus a bit more
- **SGL**: Low level python first graphics api
- **SLGU**: Thin wrapper around SGL with basic concept of structured buffers that have associated gradients and modules designed for differential calls
- **Copper**: Benedict's new experiments in creating a simpler mechanism to call Slang functions from Python, sitting on top of Falcor

Outside of these, we've got projects that can be divided up in a few different ways - some use Falcor, some use SGL, some just use SlangTorch with PyTorch. These divide up further with various approaches to forward/backward processes and PyTorch vs custom training mechanisms. Looking at the Slangtorch examples (includes learned mip maps), neural textures, neural materials, difftrace, gaussian splats and falcor ray tracers gives a wide range of examples of problems solved, and how code has 'ended up' being written under certain constraints.

Having reviewed the various options, copper represents the best high level approach in creating a clean and usable interface from Python to Slang. I've called this kernel functions for now, as its too boring a name to possibly use, so it won't accidently end up being stuck with!

See Benedikt's write up of copper for more info: https://gitlab-master.nvidia.com/bbitterli/Falcor/-/tree/copper2/scripts/internal/NeuralAppearance/python/copper2?ref_type=heads


<b>NOTE: This doc is work in progress! :) </b>


# Calling kernels and batching

Derived from **Copper**, Benedikts observation, which we all generally agree with, is that the boiler plate involved in writing explicit compute kernels need to go. A user should be able to write this function:

```
// xx.slang
float add(float x, float y)
{
    return x+y
}
```

And call it from Python, whilst being allowed to pass in either a constant or an array-like structure for either x or power, and automatically be returned either a single float or an array-like structure accordingly. i.e.

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

Of course, in GPU/Python world rather than lists we'd be likely dealing with structured buffers, tensors or numpy arrays.

## in / out / inout Parameters

Slang supports the full range of in, out and inout parameters. These would be supported in the API exactly as how one would expect them to operate:
- in: works as you'd expect
- out: writes back result (as you'd expect) - would fail to write to broadcast value
- inout: again, as you'd expect

We have discussed whether function return values should actually be forbidden, with results _only_ returnable via an `out` parameter. This has considerable benefits both in terms of matching PyTorch auto diff, and reducing ambiguities in how values should be returned in more complex batching situations. That said, in many cases this is an unnecesary restriction and would be nice to avoid.

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

If the above return none-mutating state, it would also allow users to preconfigure their calls

```
#configure once
configured_add_func = add_func
    .set({})            
    .options({})        
    .typeconformance({})

#call multiple times
configured_add_func.call(args1)
configured_add_func.call(args2)
```

This also offers up oppurtunities to trigger the call in other ways, such as adding it to an SGL command list:

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

Thus far I am inclined to suggest we leave forward differentials out of the Python api until required - they can of course still be calculated within a slang program, however I'm concerned that by attempting to wrap a Pythonic api around both forward and backward differentials we'd end up creating confusion unnecesarily, especially given 'forward' typically refers to the forward feedback step of a neural network in the context of machine learning frameworks.

## Explicit batched types

Whilst the earlier ability to pass pure lists in as batched parameters is convenient, it has the potential to cause confusion when, for example, calling a function that takes an array or even just multi dimensional types such as vectors. 

To handle this, we _could_ implement a model similar to PyTorch's `is_tensor_like`, and add a `kf.is_batchable_like`, along with a default `Batchable` type. Thus to support batching on a list, at a minimum you would be required to wrap it in a Batchable:

```
a = 10
b = Batchable([1,2,3])
result = add(a,b)
```

As the `is_tensor_like` would return true for any tensor-like type, this would also be valid (and probably more common):

```
a = 10
b = torch.Tensor([1,2,3])
result = add(a,b)
```

I am currently undecided on this point - it feels clean initially, but raises the question of what types should be 'batchable'. A numpy array? A structured buffer? If so, why these types but not a list? Perhaps if we want batching (and accordingly, broadcasting) to be explicit, it shouldn't be type based but use some other parameter mechanism?

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
    .threads(256,256)
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

A final note there is the significant parameter block containing the whole scene. In theory it could be passed to the function as well, however a user may not wish to structure the code in this highly functional way, and an API shouldn't force users to a certain way of thinking unless absolutely necessary.

# Complex Types / Type Marshalling

Thus far we've focussed on simple types like 'float', however we need to support the full range of types that Slang supports. SGL already contains extensive type marshalling support (though needs some additions), so is well suited to the problem. For this section, we can assume that basic process of converting a python dictionary to/from a slang structure is solved (as it pretty much is in SGL). That is, it is currently possible in SGL to call:

```
    computekernel.dispatch(
        thread_count=[N, 1, 1],
        vars={"processor": {"a": buffer_a, "b": buffer_b, "c": buffer_c}},
    )
```

And it will 'just work'. I believe we haven't yet added this process to all the API, and the reverse is less flushed out, but it should exist in SGL and is not too complex (albeit a little tedious!) to get going!

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

Question: Would we need to support _outputs_ having grads, i.e. if this was part of a larger backwards propagation chain?

## Tensors / AOS / SOA

When applied to machine learning, it is unlikely that you'd have a large array of Python 'ray' objects / dictionaries. Rather, you'd represent the positions / directions in tensors. This could be done in various ways

As an array-of-structures of various layouts:
- 1*1D tennor, completely flat, [ pos.x,pos.y,pos.z,dir.x,dir.y,dir.z, repeat for more rays ]
- 1*2D tensor, flat entry per ray, [ [pos.x,pos.y,pos.z,dir.x,dir.y,dir.z], ... ]
- 1*3D tensor, 2 vectors per ray, [ [[pos.x,pos.y,pos.z],[dir.x,dir.y,dir.z]], ... ]

Or a more structure of arrays form:
- 2*1D tensor, [ [pos.x,pos.y,pos.z], ... ],  [ [dir.x,dir.y,dir.z], ... ]

Plus a million other combinations of varying degrees of flatteness and AOS vs SOA mappings!

Whilst we could attempt to get **really** clever here, my inclination is to start simple and allow:

```
positions = Tensor([1,1,1], [2,2,2], ... )
directions = Tensor([0,0,1], [0,1,0], ... )
colors = Tensor()
raytrace.call(rays={'pos': positions, 'dir': directions}, colors=colors)
```

We could take this to the extreme, and attempt to allow 'auto flattening' to the correct shape, or digging down deeper into structures so pos.x, pos.y, and pos.z could all be separate tensors. I think some of this will be necessary and fall out naturally, but we should take baby steps - it has the potential to grow into a can of worms. Certainly we should enforce dimensionality matching, even if we dive into clever typing straight off.

# Background / thought process / brain dumps

<b>Note: Everything below is bits of the thought process / brain dumps that fed into the above design. They're kept for posterity / reference now, but not key to the API design.</b>

### Differentials

The interface to request differentials is tricky. I believe strongly that we need to give users the ability to request them without requiring PyTorch auto grad. The simplest starting point is to look at what slang actually generates in the case of forward and backward differentials:

```
xx.slang
//Function
[ForwardDifferentiable]
[Differentiable]
float myFunc(float a, float x)
{
    return a * x * x;
}

//Forward derivative
DifferentialPair<float> myFunc_fwd_derivative(
    DifferentialPair<float> a, 
    DifferentialPair<float> x);

//Backward derivative
void myFunc_backProp(
    inout DifferentialPair<float> a, 
    inout DifferentialPair<float> x, 
    float dResult);    
```

If we were to follow the same line of thinking thus far, the logical (though ugly) approach to making this accessible as a kernel function would be

```
yy.py
a = 3
x = 2
res = myFunc.call(a,x)
#res -> 12

#Use forward differentiation
dout_by_da = myFunc.forward.call(diffpair(a, 1), diffpair(x, 0))
dout_by_dx = myFunc.forward.call(diffpair(a, 0), diffpair(x, 1))
#dout_by_da.d -> 4
#dout_by_dx.d -> 12

#Use backwards differentiation
dout_by_da = diffpair(a)
dout_by_dx = diffpair(x)
myFunc.backward.call(dout_by_da, dout_by_dx, res)
```

This, however is not particularly user friendly, Python like, or remotely similar to how PyTorch operates!

To begin, I would propose we leave the forward option out of this convenience api (it can still be accessed via slang obviously), especially as it will cause extra confusion given in PyTorch world 'forward' is the 'feed forward' step.


In the above example, dealing with scalar, immutable values, there still isn't a lot of improvement could be done. However once dealing with lists of values, the PyTorch '.grad' model can be asily applied:

```
yy.py
a = [3]
x = [2]
res = myFunc.call(a,x)
#res -> [12]

#Use backwards differentiation
myFunc.backward.call(a, x, res)
#a.grad -> [4]
#x.grad -> [12]
```

If use of scalars was really desired, they could be wrapped in some minimal 'gradient supporting' type:

```
yy.py
a = DiffFloat(3)
x = DiffFloat(2)
res = myFunc.call(a,x)
#res -> 12

#Use backwards differentiation
myFunc.backward.call(a, x, res)
#a.grad -> 4
#x.grad -> 12
```

At that point, we could also introduce the 'needs grad' approach taken in PyTorch, either inferring it directly from the type provided, or checking for a 'needsgrad' property internally, and as in Copper only generate gradients for variables that're useful:

```
yy.py
a = DiffFloat(3)
x = 2
res = myFunc.call(a,x)
#res -> 12

#Use backwards differentiation
myFunc.backward.call(a, x, res)
#a.grad -> 4
```

This basic layout feels correct to me, and much more akin to how most Python libraries operate.

My knowledge of PyTorch at the moment isn't strong enough to know how we'd integrate that with PyTorch if-and-only-if we wanted to (i.e. PyTorch might not be there, or just because PyTorch is, doesn't mean calling backward means we want autograd doing stuff).

### Gradient accumulation

With the system above, even if we wanted to ignore gradient accumulation, we couldn't. Code such as the following would introduce an ambiguity:

```
xx.slang
[Differentiable]
float myFunc(float a, float x)
{
    return a * x * x;
}

yy.py
a = DiffFloat(3)
x = Tensor([1,2,3], needsgrad=True)
res = myFunc.call(a,x)
#res -> [3,12,27]

#Use backwards differentiation
myFunc.backward.call(a, x, res)
#a.grad -> ???
#x.grad -> [6,12,18]
```

Following the earlier batching rules we've assumed it's valid to pass a and x into backward to get resulting gradients. However whilst it was possible to batch differently during the call, there is now no clearly defined value for a.grad. You could argue:
- a.grad should be d/d_a for all values of res (i.e. a become list)
- a.grad should be an accumulation of d/d_a (i.e. remain a scalar)

A database 'group' query often hits a similar problem, and typically treats the first option as a special case of accumulation - 'push' accumulation, in which every result is added to list rather than accumulated through some mathematical function. 

With that in mind, _if_ we wanted to the logical way to clear up the ambiguity would be some form of accumulator flag:

```
a.accumulator = 'sum'
myFunc.backward.call(a, x, res)
#a.grad -> sum of dres_by_da

a.accumulator = 'push'
myFunc.backward.call(a, x, res)
#a.grad -> list of dres_by_da

a.accumulator = 'first'
myFunc.backward.call(a, x, res)
#a.grad -> first dres_by_da
```

This has the neat benefit that it can be applied to both a AND x, with the batch dimensionality defined by res. For example, even if the input 'x' was a list, we could still ask gradients to be accumulated into a single scalar.

## Benedikit thoughts (18/07/24)

### Kernels/batching:
My gut feeling is that there should be a pretty explicit signal of user intent about what should be batched. In the Falcor copper version (Falcopper...?) this is basically decided by checking if an input is_tensor_like. This spec is a bit more liberal and allows e.g. lists to be batched as well (does that extend to all iterables?). I worry this might be a bit too accepting and would allow committing some weird bugs when code gets more complex. I think the batched list case is sufficiently rare---input data is either uniform or you have a lot of it and a list would be a poor choice then---that you can require things to be wrapped in a tensor-like object if they should be batched. Then you can reserve lists for describing slang types like arrays, vectors or matrices and throw more detailed errors when users make mistakes

In "More complex batching" you actually implemented something I was debating for a while, which is batching things with different dimensionality (i.e. shape [2, 3] and [3] batched together make an output shape [2, 3]). In the "array of numbers" world of python libraries, there are established rules for how this should work. Here's the torch page about broadcasting rules: https://pytorch.org/docs/stable/notes/broadcasting.html . This can be really useful, e.g. pass a tensor of shape [256, 1] and one of shape [1, 256], and the output is returned with shape [256, 256]. OTOH, it can lead to some amazing bugs when you accidentally swap some arguments, or they're just not the shape you thought they were, and the framework happily accepts it and mashes the shapes together into an enormous giga tensor of failure :slightly_smiling_face: For now I was erring on being more restrictive and requiring the batch sizes of everything to either be [] (i.e. broadcast) or to be exactly identical. I do admit I like the torch broadcasting rules---they're very convenient---but they also scare me a bit
The spec doesn't touch on inout / out parameters, are you planning to prohibit those in the initial version?

Writing the differentiation spec separate from torch.autograd is good. Whatever the spec looks like, we'll offer compatibility layers for various frameworks (like torch) and wrap it into the form their autograd expects. I only know the internals of torch, but I can double check the spec stays compatible with at least that.

### Some thoughts about differentiation:
The base spec only mentions marshalling of scalars/basic types, but differentiation already asks for some basic structure support. DifferentialPair is a struct with a .p and .d field, and those would hold different tensors (they come separately from the forward or the backward direction). When you go to higher derivatives, you even start nesting differential pairs, and .p and .d might not even nest to the same depth... My main motivation for dealing with struct marshalling early was that differentiation could be handled without needing special casing for DifferentialPairs (which would end up looking like struct marshalling anyway)

I understand why the diffpair API is required in slang, but from a user perspective it's quite painful, and every time I do non-trivial differentiation I have to go back and read the slang spec again to understand what it expects. I wonder if we can make it easier to use from the python side; I have some thoughts, but I'm also not sure this is needed, since most of the time it will be hidden behind some autograd framework anyway.

Outputs from kernel calls need to be tensor-like if you want them to be tracked by the autograd from other frameworks. Doing something like z = add(x,y) with everything a python float would break, invisibly---the user could go on to pass z to other things, and then invoke e.g. backward down the line expecting everything to propagate, but the graph is disconnected at z in a way that can't be detected. You can't even abstract it in a way where the base layer talks scalars and somewhere higher in the abstraction (e.g. the torch layer) wraps it into tensors; as soon as you unwrap it into a scalar / python type you lose tracking. For that reason I was forcing return values and out / inout parameters to always be tensors

### About accumulation:

Gradient accumulation warrants being extremely careful. The easiest thing for now is to just not do any of it: Accumulation is none of copper's business; if you want gradients of a parameter, it has to be batched with no broadcasting whatsoever, and you will get back a batch's worth of gradients. On the torch side, before invoking the kernel you would then e.g. call expand to turn your model parameters into a batch's worth of copies of the model parameters, and  torch.autograd will perform the sum for you when it backprops through the expand call. This will work for basic things, but when you have many parameters then this becomes impractical. Say you have a 4k texture and a batch size of 64k; may I offer you a giga tensor of 4096 * 4096 * 65536 gradients?

The next step is to support accumulation and use the same rules as autograd frameworks would. Mathematically, differentiating a broadcast always becomes a sum. We know when the user broadcasts an input, and so internally we know we have to do accumulation when you call backward. With some care this should be doable and intuitive. There might be cases when an input is broadcast only over some dimensions; e.g. in the "More complex batching" example, y=[1,2,3] is broadcast over dimension 0 and batched over dimension 1. Mathematically you would sum gradients over dimension 0 and get back a 1D tensor of gradients, and this is what happens in any current autodiff framework. We could do the same, but I'm wary of the bug attack surface---it's actually the main reason I decided against allowing more liberal broadcasting rules for now.