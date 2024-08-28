# Proposed broadcasting rules

## Shape types

For a given slangpy function call, we define an N dimensional **call shape, CS** and call index, *call_idx*. This shape is always inferred regardless of whether explicit mappings were provided.

The kernel shape is analogous to the 'num threads' used to disaptch a classic compute kernel, and the call index is analogous to the thread_id passed to the entry point.

Each slang type has an implicit shape, (the **type shape, TS**), that can be determined via reflection. For example:
- Scalar consumes 1D (though some internal work has to occur to treat as `vector<T,1>`)
- `vector<T,N>` consumes 1D
- `matrix<T,N,M>` consumes 2D
- `Texture2D<T>` consumes 3D
- `StructuredBuffer<float4>` consumes 2D
- `StructuredBuffer<float4x4>` consumes 3D
- `StructuredBuffer<OpaqueStruct>` consumes 1D
- `float4[]` consumes 2D
- `Slice(3,float2)` consumes 5D (more later)

An argument or return value of a kernel function all consume the dimensions required by their type from the smallest first. Note: For the purposes of this document, a return value is simply a special case of an output argument.

The remaining dimensions define the **argument shape, AS**, and argument index, *arg_idx*

For example, an argument of type ```float4[]``` is passed a tensor of shape ```(100,16,4)```:
- Type shape = ```(16,4)```
- Argument shape = ```(100)```

The challenge of broadcasting is ultimately working out how to express and generate the mapping *call_idx* -> *arg_idx* for all arguments.

## Shape inference

Slangpy must resolve both the call shape, and the shape of any arguments/return value in order to dispatch. 

Arguments are resolved as follows:
- An ```in``` or ```inout``` param takes the shape it was passed
- An ```out``` or ```return``` param can be passed
    - A pre-allocated buffer, which defines the shape according to the same rules
    - A reference to *fill in*, allowing slangpy to decide on the output shape
    - An explicit dimensionality
    - An explicit shape

If a user attempts no override of dimension mapping or vectorisation, basic numpy rules are applied:
- All arguments with a concrete shape are used to calculate the call shape using numpy rules (basically simple broadcasting, where none-1D dimensions must match in size)
- If the numpy rules are breached, it is an error and raises an exception
- Any undefined argument shapes are assigned the call shape, allowing the underlying buffer size to be calculated by combining argument+type shapes

## Basic examples

### Simple function call

This basic call has only input arguments and a return value. As such, there is no situation in which any ambiguity can arise, and it is always possible to infer a concrete call + result shape.

```
xx.slang
float dot(float3 a, float3 b) 
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z; 
}
```

```
yy.python

# Type shapes for arguments and return value
# a_TS = (3)
# b_TS = (3)
# r_TS = (1)

# simple scalar call
# a_AS = ()
# b_AS = ()
# CS   = ()
# r_AS = ()
dot(float3, float3)

# error: no implicit conversion of concrete types
dot(10.0, float3)

# broadcast a
# a_AS = ()
# b_AS = (100)
# CS   = (100)
# r_AS = (100)
dot(float3, tensor(100,3))

# broadcast b
# a_AS = (100)
# b_AS = ()
# CS   = (100)
# r_AS = (100)
dot(tensor(100,3), float3)

# error - b's lower dimension can't be consumed by vector 3
# a_AS = (100)
# b_AS = <error>
dot(tensor(100,3), tensor(100,1))

# error - numpy broadcast impossible
# a_AS = (100)
# b_AS = (1000)
# CS   = <error>
dot(tensor(100,3), tensor(1000,3))

# bigger broadacst
# a_AS = (100)
# b_AS = (1000,100)
# CS   = (1000,100)
# r_AS = (1000,100)
dot(tensor(100,3), tensor(1000,100,3))
```

### Output parameter

The call adjusted to use an out parameter is still concrete, however the user now has to provide outputs to populate, and if fixed size outputs are provided, race conditions can occur. 

```
xx.slang
void dot(float3 a, float3 b, out r) 
{ 
    r = a.x * b.x + a.y * b.y + a.z * b.z; 
}
```

```
yy.python

# Type shapes for arguments and return value
# a_TS = (3)
# b_TS = (3)
# r_TS = (1)

# Use slangpy scalar ref to receive floating point output
# a_AS = ()
# b_AS = ()
# CS   = ()
# r_AS = ()
output = ScalarRef(0.0)
dot(float3, float3, output)

# Race condition on r detected as fixed shape output
# If error explicitly ignored, all calls will just
# attempt to write to r 
# a_AS = (100)
# b_AS = ()
# CS   = (100)
# r_AS = ()
output = ScalarRef(0.0)
dot(tensor(100,3), float3, output)

# Undefined buffer output allows 
# slangpy to assign it the call shape
# a_AS = (100)
# b_AS = ()
# CS   = (100)
# r_AS = (100)  [inferred]
output = BufferOutput(t=float3)
dot(tensor(100,3), float3, output)

# Buffer with dimensionality constraint causes 
# error due to undefinable dimensions
# a_AS = (100)
# b_AS = ()
# r_AS = (?,?,100) [error: undefinable dimensions]
output = BufferOutput(t=float3, dimensions=3)
dot(tensor(100,3), float3, output)

# Buffer with dimensionality constraint simply
# helps with error checking
# a_AS = (100)
# b_AS = ()
# r_AS = (100) [inferred]
output = BufferOutput(t=float3,dimensions=1)
dot(tensor(100,3), float3, output)
```

### None-fixed-shape arguments

In this example I've introduced a concept of an 'ND slice'. This is designed to act as an interface that can wrap being passed some subset of dimensions of a higher dimensionality structure such as a tensor or buffer. In this case the slice is a 2D container for float4s, giving it a **type shape** of ```(N,M,4)```.

Interestingly, in this context, just the fixed dimensionality provides a means to resolve any ambiguity, however we do need to figure out the optimal way to pass this 'slice' of data in to the 'read' function. The exact same logic can apply when multiple parameters are slices. I have not proven, but am fairly confident, that this logic can also be extended to out parameters using the same rules.

```
xx.slang
float4 read(int2 index, Slice<2,float4> array) 
{ 
    return array[index]
}
```

```
yy.python

# Type shapes for arguments and return value
# a_TS = (2)
# b_TS = (N,M,4)
# r_TS = (4)

# Pass a tensor of the same dimensionality, exact shape
# is inferred and no broadcast occurs
# b_TS = (100,100,4)
# a_AS = ()
# b_AS = ()
# CS   = ()
# r_AS = ()
read(int2, tensor(100,100,4))

# Broadcast the tensor across 50 indexes
# b_TS = (100,100,4)
# a_AS = ()
# b_AS = (50)
# CS   = (50)
# r_AS = (50)
read(buffer(50,int2), tensor(100,100,4))

# Call 50 times with different slices for each
# b_TS = (100,100,4)
# a_AS = (50)
# b_AS = (50)
# CS   = (50)
# r_AS = (50)
read(buffer(50,int2), tensor(50,100,100,4))

# Error: incompatible type shape for b
# b_TS = (100,100,<error>)
read(buffer(50,int2), tensor(50,100,100,5))

# Error: invalid broadcast dimensions
# b_TS = (100,100,4)
# a_AS = (50)
# b_AS = (75)
# CS   = <error>
read(buffer(75,int2), tensor(50,100,100,4))
```

## Remapping and explicit vectorizing

Most of the examples above involve basic vectorization of a function through implicit broadcasting, as defined by the numpy broadcast rules. However there are 2 additional tools we can provide for advanced use:
- Remapping the dimensions of a given argument
- Controlling how argument shapes map to / affect call shape

In a classic ML library these could be merged into 1, however I believe that with the graphics oriented pipeline in which more complex data types such as vectors, matrices and textures exist, separating them makes more sense.

# Remapping

Beginning with the simple function call:

```
xx.slang
float dot(float3 a, float3 b) 
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z; 
}
```

```
# a_AS = ()
# b_AS = (100,100)
# CS   = (100,100)
# r_AS = (100,100)
dot(float3, tensor(100,100,3))

# Error as b can not be mapped to the type shape (3)
# a_AS = ()
# b_AS = <error>
dot(float3, tensor(3,100,100))
```

To allow the user more flexibility with input dimensions, a simple resolution would be

```
# a_AS = ()
# b_IS = (3,100,100) -> (100,100,3)
# b_AS = (100,100)
# CS   = (100,100)
# r_AS = (100,100)
dot.map(b=(1,2,0)) # remap B's dimensions
   .call(float3, tensor(3,100,100))

# or using similar syntax
dot2 = dot.map(b=(1,2,0))
dot2(float3, tensor(3,100,100))

# or 1-liner
dot.map(b=(1,2,0))(float3, tensor)

# or maybe using an options structure
dot(float3, tensor(3,100,100), _options={
    'remap': {
        'b': (1,2,0)
    }
})
```

This remapping is **always** applied first, before any other shape calculations, as it is a pure transformation of the input.

# Explicit vectorizing

There are also situations in which the basic numpy rules are not enough to express complex broadcasting rules. A great example of this is in batch training:

```
xx.slang
float4 read(int2 index, Slice<2,float4> array) 
{ 
    return array[index]
}
```

```
yy.python

# Type shapes for arguments and return value
# a_TS = (2)
# b_TS = (N,N,4)
# r_TS = (4)

# User wishes to perform 1000 reads against 50 100x100x4
# inputs during batch training.
# a_AS = (1000)
# b_TS = (100,100,4)
# b_AS = (50)
# CS   = <error>
read(inttensor(1000,2), tensor(50, 100, 100,4))
```

In this example a user has a perfectly reasonable request. A concrete, unchanging set of values for argument (a), and 50 batches of (100,100,4) buffers to sample. However basic numpy rules would fail. 

```
yy.python
# Function transform adjusts call shape, thus broadcasting can work
# a_AS = (1000)
# b_TS = (100,100,4)
# b_AS = (50)
# CS   = (1000,50)
# r_AS = (1000,50)
read.vmap("(N),(M)->(N,M)")
    .call(inttensor(1000,2), tensor(50, 100, 100,4))

#or, as with map, other calling styles
read.vmap(bla)(args)

read2 = read.vamp(bla)
read2(args)

read(args, _options={'vmap': 'bla'})
```

This explicit mapping of input dimensions to output dimensions tells slangpy both how to calculate the overall call size, and how to do indexing at the kernel level. 

## Named dimensions (wip proposal)

Allowing users to name dimensions in their buffers/tensors is already becoming popular, and we can make use of it our end elegantly:
- For a .map, explicit dimension names instead of indices can be used
- For .vmap, no aliases are needed (it can just be vmap("a.dim1","b.dim2") etc)
- A user can opt-in to always specify .map and/or .vmap for some/all calls to aid in error checking

From the earlier dot product map example:

```
# a_AS = ()
# b_IS = (c,w,h)=(3,100,100) -> (w,h,c)=(100,100,3)
# b_AS = (100,100)
# CS   = (100,100)
# r_AS = (100,100)
mytensor = tensor(channels=3, width=100, height=100)
dot.map(b=('width','height','channels')) # remap B's dimensions
   .call(float3, mytensor)
```

Or the vmap example:

```
yy.python
# Function transform adjusts call shape, thus broadcasting can work
# a_AS = (s)       = (1000)
# b_TS = (w,h,c)   = (100,100,4)
# b_AS = (b)       = (50)
# CS   = (b.b,a.s) = (50,1000)
# r_AS = (b.b,a.s) = (50,1000)
tensor_a = inttensor(samples=1000, indices=2)
tensor_b = tensor(batches=50, width=100, height=100, channels=4)
read.vmap('array.batches','read.samples')
    .call(tensor_a, tensor_b)
```

