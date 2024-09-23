# Overloads and call shape resolution

This builds on the thoughts in the BroadCastShapes.md doc, now having attempted to write it in a few different ways.

This refined overview from broadcastshapes.md is the foundation of overloads. For a given slangpy function call, we define an N dimensional **call shape, CS** and call index, *call_idx*. This shape is always inferred regardless of whether explicit mappings were provided.

The kernel shape is analogous to the 'num threads' used to disaptch a classic compute kernel, and the call index is analogous to the thread_id passed to the entry point.

Most types, even vectors can be thought of as a container with an element type. The itself container has a shape, and an element type (which may also be a container) has a shape. For any type therefore, the overall shape can be defined as:

`shape = container_shape + element_type.shape`

Each slang type has an implicit shape, (the **type shape, TS**), that can be determined via reflection. For example:
- Scalar consumes 0D
- `vector<T,N>` 1D container of fixed size N, with element T
- `matrix<T,N,M>` 2D container of fixed size N,M with element T
- `Texture2D<T>` 2D container with element T
- `StructuredBuffer<float4>` 1D container with element `vector<float,4>`
- `float4[]` 1D container with element `vector<float,4>`

The job of an overload system in the case of slangpy is to take the input shapes/element types from python arguments, and match them against shapes/element types of slang parameters.

## Slang Function Declaration Parameters

If generics are not involved, a given slang function declaration gives us:
- A known **dimensionality** of each parameter
- A known **element type** of each parameter
- For some types (such as fixed size vectors), it also gives us a concrete shape

If broken down to their core, a function such as the following:

`float myfunc(float3 vec, int2[10] array, Texture2D<float> texture, Particle myparticle)`

Can be said to have parameter information:
- `vec`: element=`float`, shape=`[3]`
- `array`:
    1. element=`int2`, shape=`[10]`    
    2. element=`int`, shape=`[10,2]`
- `texture`: element=`float`, shape=`[?,?]`
- `myparticle`: element=`Particle`, shape=`[]`
- `_retval`: element=`float`, shape=`[]`

Note how the array can be broken down in 2 steps to its core type + combined shape.

Even in the event that the user wishes to provide a nested type to the particle parameter, the struct's fields provide similarly concrete dimensionality and element type.

So, for a none-generic slang function declaration, really the only ambiguities can be the size of given dimensions. Dimensionality and type are always fully resolvable.

## Python Function Call Arguments

From the python side, inputs are a lot more vague for various reasons:
- A user can simply specify 'None' if they don't care about an output
- A user may not have specified a return value buffer, expecting it to be allocated
- Nested dictionaries themselves have no associated python type
- The values inside nested dictionaries may have different shapes
- Structured buffers may only have a stride, not a type
- The dimension sizes for some custom arguments (such as random numbers) is defined by the call shape

Additionally, it's critical the Python side is seen as 2 very independent phases:
- Potentially expensive kernel generation / compilation 
- Extremely low overhead kernel execution

For performance, kernel re-use is key - we do not want to have to generate a new kernel for different sized buffers. As a result, at the point of generation we must assume the dimensionality of a buffer is concrete but not its size.

This complex semi-psuedocode call shows a host of problems:

```
# Define various contains
vectors = NDBuffer(eltype=float, shape=(10,10,3))
array = StructuredBuffer(elsize=8*10, elcount=10)
texture = Texture2D(eltype=float, width=128, height=256)
positions = NDBuffer(eltype=float3, shape=(10))
velocity = float3(0,1,0)

# Call function, passing nested type for particle and expecting 
# result to be allocated automatically
result = myfunc(vectors, array, texture, {
    'position': positions,
    'velocity': velocity
})
```

From a purely python side, at kernel generation time, the concrete information we have is:

- `vectors`: element=`float`, shape=`[?,?,?]`
- `array`: element=`?`, shape=`[?]`
- `texture`: element=`float`, shape=`[?,?]`
- `particle`:
    - `position`: element=`float`, shape=`[?,3]`
    - `velocity`: element=`float`, shape=`[3]`
- `result`: element=`float`, shape=`call shape`

If we want to push resolution fully into Slang whilst maintaining the current feature set, this is roughly the data it would need to understand.

## Pairing Slang + Python / kernel gen

The interesting observation is that this process has much more in common with the constraint based solving in a generic compiler than classical overload resolution, as the shape of some parameters can be defined through resolution of others.

A first pass compatibility check for overload resolution can just match leaf element types against leaf element types. If an overload has any elements that can not be paired (including within nested structures), it is by definition not a candidate. 

If the user has provided additional typing information (either explicitly, or using instance lists) this can also be applied during compatibility checks.

Once a potential overload is found, the number of dimensions that correspond to its Slang parameters are *consumed*. This leaves the *argument shapes* - i.e. those that define the call shape, which is needed for kernel generation:

- `vectors`: element=`float`, shape=`[?,?]`
- `array`: element=`?`, shape=`[?]`
- `texture`: element=`float`, shape=`[]`
- `particle`:
    - `position`: element=`float`, shape=`[?]`
    - `velocity`: element=`float`, shape=`[]`
- `result`: element=`float`, shape=`[?,?]`

The 3 key pieces of information required for kernel gen are now known:
- The uniforms (defined predominantly by python type, with some use of slang when element types are ambiguous)
- The call shape (inferred from above)
- The load/store target types (defined by slang parameters)

It is *sometimes* possible to reject an overload due to shapes at this point. For example, if a `vector<float3>` is being passed to a `float[10]`, the relevant dimensions are concrete and incompatible. However in most situations, shape compatibility can't be resolved up front.

For robust overload resolution, slangpy must run the above logic against **all** potential overloads, and only succeed if there is **exactly 1** match.

## Executing the kernel

At call time, all input containers have concrete shapes, which allows decisions about what is to be broadcast and what the actual call dimensions are.

Inserting 'N' for a broadcast dimension, and X,Y for shapes dimensions:
- `vectors`: element=`float`, shape=`[X,Y]`
- `array`: element=`?`, shape=`[N,Y]`
- `texture`: element=`float`, shape=`[N,N]`
- `particle`:
    - `position`: element=`float`, shape=`[N,Y]`
    - `velocity`: element=`float`, shape=`[N,N]`
- `result`: element=`float`, shape=`[X,Y]`

Then at point of kernel call, this is resolved further to:
- `vectors`: element=`float`, shape=`[10,10]`
- `array`: element=`?`, shape=`[N,10]`
- `texture`: element=`float`, shape=`[N,N]`
- `particle`:
    - `position`: element=`float`, shape=`[N,10]`
    - `velocity`: element=`float`, shape=`[N,N]`
- `result`: element=`float`, shape=`[10,10]`

In this case, all parameters agree on X and Y. If not, numpy broadcast rules would have been breached and the user is informed of an error.

With both overall call shape, and the individual shape information for each argument, it is now possible to populate uniforms that allow for any combination of loads/stores that the container types support.

## Remapping

Whilst syntax can vary, the ability to remap dimensions of input shape to specific dimensions of the call shape makes everything a **lot** more complex!

In the simplest situation, the user provides remapping for argument shapes that map to dimensions of the call shape that would exist in the absence of any remapping. For example, in the earlier case, the call shape is determined to be of dimensionality 2. If no argument dimensions are mapped to call dimensions outside of the range 0-1, the impact upon code gen will be 0 - all changes can be accounted for with uniforms.

The first stage of complexity arises with a more complex remapping commonly seen with batch training:

```
xx.slang
float multiply(float a, float b) {
    return a * b;
}

yy.py
a = NDArray(eltype=float, elcount=100)
b = NDArray(eltype=float, elcount=50)

res = module.map(a=(1,)).multiply(a,b)
```

Results in
- `a`: element=`float`, shape=`[100]`
- `b`: element=`float`, shape=`[50]`
- `result`: element=`float`, shape=`call_shape`
- `call_shape`=`[?]`

With the 'map' modification, this would pass kernel gen fine, with a call shape of `[?]`, but fail at call time on finding the dimension sizes are not compatible.

The broadcast projects (a) to the 2nd dimension, giving
- `a`: element=`float`, shape=`[100]`, mapped=`[?,100]`
- `b`: element=`float`, shape=`[50]`, mapped=`[50,?]`
- `result`: element=`float`, shape=`call_shape`
- `call_shape`=`[?,?]`

i.e. the simply mapping of (a) has adjusted the call shape, and thus had impacts for how all other arguments are mapped / broadcast.

This introduces a question though of how mapping should work when the inputs are float3s instead of floats. To obtain the same result, the mapping command becomes more complex:

```
xx.slang
float multiply(float3 a, float3 b) {
    return a * b;
}

yy.py
a = NDArray(eltype=float3, elcount=100)
b = NDArray(eltype=float3, elcount=50)

res = module.map(a=(1,2), b=(0,2)).multiply(a,b)
```

To handle this situation, we have to go right back to pre-element-strip and analyse the full shapes of the python types.

Results in
- `a`: element=`float`, shape=`[100,3]`, mapped=`[?,100,3]`
- `b`: element=`float`, shape=`[50,3]`, mapped=`[50,?,3]`
- `result`: element=`float`, shape=`call_shape`

Which can then be stripped and finalized to:
- `a`: element=`float`, shape=`[100,3]`, mapped=`[?,100]`
- `b`: element=`float`, shape=`[50,3]`, mapped=`[50,?]`
- `result`: element=`float`, shape=`call_shape`
- `call_shape` = `[?,?]`


This becomes even more complex once sub-kernel level indexing is introduced, such as interpretting a float tensor as a float3 input.

```
xx.slang
float3 multiply(float3 a, float3 b) {
    return a * b;
}

yy.py
a = NDArray(eltype=float, shape=(3,10,20))
b = NDArray(eltype=float3, shape=(10,20))

res = module.map(a=(2,0,1)).multiply(a,b)
```

This mapping is absolutely valid, and potentially very desirable when image processing kernels are used. However the mapping of (a) now actually contains a dimension index that will sit **outside** of the call shape! 

For this call, we must go right back to before element shapes are stripped base on slang inputs.

- `a`: element=`float`, shape=`[3,10,20]`, mapped=`[10,20,3]`
- `b`: element=`float`, shape=`[10,20,3]`, mapped=`[10,20,3]`
- `result`: element=`float`, shape=`call_shape`

Element sizes can now be stripped safely, and the final call shape be calculated:

- `a`: element=`float`, shape=`[3,10,20]`, mapped=`[10,20]`
- `b`: element=`float`, shape=`[10,20,3]`, mapped=`[10,20]`
- `result`: element=`float`, shape=`call_shape`
- `call_shape`=`[?,?]`

The call shape is now valid, and the kernel will be called the correct number of times, but the slang side types need to be able to *invent* an extra dimension when sampling `a` - it needs to treat a call id of `[10,20]` as 3 calls with ids `[10,20,0]`, `[10,20,1]` and `[10,20,2]`. Only then can it correct load from `[0,10,20]`, `[1,10,20]` and `[2,10,20]`


## Overload resolution process

The upshot of all this is that the overload/kernel gen process has the following steps for all potential overloads:
1. Break down all Python arguments to their leaf element type and dimensionality
2. Do the same with Slang parameters and bind the 2 together, matching element types
3. Where necessary, infer ambiguous element types from Slang parameters
4. Calculate mapped dimensionality for each input
5. Consume Slang parameter dimensionality 
4. Generate kernel

The call process is then
1. Calculate shape, and then mapped shape for all inputs
2. Validate broadcasting rules
3. Setup uniforms with enough information to apply broadcasting+mapping rules in kernel
4. Allocate buffers to match call shape where necessary
5. Dispatch kernel

