# kernelfunctions



## Overview

Currently we have several implementations of systems that integrate python with slang, and various projects sitting on top of them:
- **Slang Torch**: CUDA based python project for compiling and calling CUDA functions written in slang and using them with pytorch
- **Falcor**
- **SGL**: low level python graphics api
- **SLGU**: Early experimental wrapper around SGL with basic concepts of structured buffers that have associated gradients
- **Copper**: Benedict's new experiments in creating a simpler mechanism to call Slang functions from Python, sitting on top of Falcor

Outside of these, we've got projects that can be divided up in a few different ways - some use Falcor, some use SGL, some just use SlangTorch with PyTorch. 

Having reviewed the various options, copper represents the best approach in creating a clean and usable interface from Python to Slang. I've called them kernel functions for now, as its too boring a name to possibly use, so it won't accidently end up being stuck with!

See Benedikt's write up of copper for more info: https://gitlab-master.nvidia.com/bbitterli/Falcor/-/tree/copper2/scripts/internal/NeuralAppearance/python/copper2?ref_type=heads

### Calling kernels and batching

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

### More complex batching

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

### Remembering the basics!

Before getting all clever, it's important to remember that we want to make the simpler cases easy as well - partly as that's just useful, but also when the fancy functionality of a library doesn't quite cut it, users need to go back to writing simpler more bespoke code themselves. For example:

```
xx.slang
void fill(int2 pixel, float3 color, RWTexture2D<float3> texture) {
    texture[pixel] = color
}
```

Should provide a mechanism for setting pixel to a range of values. 

### Simple ray tracer example

Before worrying about batching or gradients, its worht looking at a very basic ray tracer kernel:

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

Ideally this could be made to work trivially, just by 'calling' the main function, but then enhanced if need be. For example, a v1 call might be:

```
yy.py
main.set({
        'g_scene' = bla,
        'g_output' = foo,
        'g_frame' = 10
    })
    .options(threads=[16,16])
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
raytrace.set(g_scene=bla)
    .call(
        frame=10,
        pixel=<some indication it is a range the size of MyTexture??>,
        res=MyTexture
    ) 
```

In this case though a key optimization has been lost - the original ray tracer quite deliberately used cache coherent blocks of 16x16 pixels. This suggests still we'd want the ability to add additional options before the call:

```
yy.py
raytrace.set(g_scene=bla)
    .options(threads=[16,16])
    .call(
        frame=10,
        pixel=<some indication it is a range the size of MyTexture??>,
        res=MyTexture
    ) 
```

A final note there is the significant parameter block containing the whole scene. In theory it could be passed to the function as well, however a user may not wish to structure the code in this highly functional way, andan API shouldn't force users to a certain way of thinking unless absolutely necessary.

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



