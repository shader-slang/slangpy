# Slangpy API Review

These proposed modifications to slangpy have the following goals:
- Fully expose the functionality of Slang.
- Minimize overhead in supporting future Slang features.
- Ensure seamless integration with future fusion implementations.
- Provide clear, easy-to-use, and explicit vectorization.
- Allow for implicit vectorization where appropriate.

## Key Definitions
To ensure clarity, the following terms are defined:
- **Vectorization**: The process of converting a function from its scalar form to one designed to be called multiple times from a kernel.
- **Mapping**: How the dimensions of an argument correspond to the dimensions of the kernel.
- **Typing**: The type of an argument loaded/stored within the kernel and passed to the function.
- **Resolution**: The process of taking a set of Python arguments and producing a fully resolved function, where the mapping and typing of all arguments are known.

A fundamental design principle is that Python should **never** inspect the Slang function signature during resolution.

## Explicit vectorization

In the most basic implementation, all arguments must either:
- Not be vectorized.
- Have explicitly defined mappings, allowing type inference.
- Be explicitly typed, allowing mapping inference.

For example, given the following Slang function:

```slang
float3 tonemap(float3 color, float filmic, float saturation) {
    ...
}
```

This function could be called in Python in several ways:

```python
# No vectorization
val = float3(...)
result = tonemap(val, 1, 0.5)

# Vectorization with explicit mapping
val = make_texture2d_float3()
result = tonemap(vmap(val, (0,1)), 1, 0.5)

# Vectorization with explicit typing, which provides implicit mapping
val = make_texture2d_float3()
result = tonemap(vcast(val, m.float3), 1, 0.5)

```

For scenarios where performance is critical, pre-vectorization is also possible:

```python
# Pre-vectorization with explicit mapping (typing is implicit)
tonemap2d = tonemap.declare(vmapping((0, 1)))
val = make_texture2d_float3()
result = tonemap2d(val, 1, 0.5)

# Pre-vectorization with explicit typing (mapping is implicit)
tonemap2d = tonemap.declare(m.float3)
val = make_texture2d_float3()
result = tonemap2d(val, 1, 0.5)
```

In these examples, only the first parameter is vectorized. In more complex cases, a combination of mappings and typings can be applied:

```slang
// Updated tonemap function with an output parameter
void tonemap(float3 color, float filmic, float saturation, out float3 result) {
    ...
}
```

```python
# Explicit mapping for 'val' and explicit typing for 'result'
val = make_texture2d_float3()
result = make_texture2d_float3()
tonemap(vmap(val, (0, 1)), 1, 0.5, vcast(result,m.float3))

# Pre-vectorized version
tonemap2d = tonemap.declare(vmapping(0, 1), None, None, m.float3)
```

### Key Points
- The Python 'call' operator invokes the function. Pre-vectorization requires an explicit `declare` call, which improves performance and readability in simple cases, but has implications for the future fusion syntax.
- The `vcast` syntax is a function call taking value and type, as apposed to constructor like syntax. This is to remain compatible with a future fusion system.
- A `vdim` operator could be added as a shorthand for `vmap` in which the dimensions are ordered (i.e. `vdim(val,3) == vmap(val,(0,1,2))`), which is a very common case
- Unlike Jax, argument mapping occurs within the function call, aligning with the casting syntax and avoids long strings of `None` for functions where only the last argument is vectorized.
- Note that if desired, we could use the Jax style syntax, provided we switched to it for both mapping and casting. However it does not lend itself well to large function calls with nested arguments.

The core idea here is that Python can fully resolve the vectorization without needing to know the details of the `tonemap` function.

## Implicit vectorization

To reduce boilerplate code and improve iteration times, implicit vectorization is introduced. This works like an implicit type cast during the resolution step.

Consider the same `tonemap` example:

```slang
float3 tonemap(float3 color, float filmic, float saturation) {
    ...
}
```

With implicit vectorization, it can be called in Python as:

```python
# Implicitly cast a float3 texture to a float3
val = make_texture2d_float3()
result = tonemap(val, 1, 0.5)
```

This approach allows the user to improve performance without adding Python-side decorators. The call to `tonemap` is now on par with a pre-vectorized version from V1 in terms of performance.

For this to work, Slang must handle implicit casting during the generation process, similar to:

```pseudocode
# Returns [float3, float, float]
slang_session.resolve(tonemap, Texture2D<float3>, float, float)
```

This could involve returning a `FunctionReflection` object if needed. The main design decision is how the custom *operators* are defined that tell slang what conversations are valid.


### Potential confusion with float tensors

Implicit conversions can boost productivity but may also introduce bugs. For example, casting tensors to vectors could cause issues if not carefully managed. Slang code generation isn't bound to fixed shapes—only dimensionalities. As a result, the following calls produce the same kernel and only differ in the uniforms passed:

```python
val = NDBuffer(element_type=float, shape=(100, 3))
result = tonemap(val, 1, 0.5)

val = NDBuffer(element_type=float, shape=(100, 2))
result = tonemap(val, 1, 0.5)

val = NDBuffer(element_type=float, shape=(100, 1))
result = tonemap(val, 1, 0.5)
```

We could create an implicit operator to convert `NDBuffer<float, N>` to `float3`, provided the `NDBuffer` can load a `float3`. However, this could lead to issues without additional error handling.

Alternatively, we could require explicit casting:

```python
val = NDBuffer(element_type=float, shape=(100, 3))
result = tonemap(vcast(val,m.float3), 1, 0.5)
```

## Fusion

Fusion introduces a third form of implicit mapping—if an argument is the result of a vectorized function, its mapping is by default inherited from that function.

The challenge with fusion lies in designing an API that allows calls to be deferred until fusion occurs, ensuring both high performance and support for pre-vectorization. The proposed solution is for the user to provide a fusable function, which is then wrapped with a call to `fuse`:

```slang
float3 sample_texture(float2 uv) {...}
float3 tone_map(float3 color, float filmic, float saturation) {...}
```

```python
# Define a Python function
def tone_map_sample_func(uv):
    return tone_map(sample_texture(uv), 1, 0.5)

# Create a fused version of the function
tone_map_sample = fuse(tone_map_sample_func)

# The fused function can be called as normal
buffer = make_buffer_of_float2s()
results = tone_map_sample(buffer)

# Or it can be pre-vectorized
tone_map_sample_2d = tone_map_sample.declare(vmapping(0,))
tone_map_sample_2d = tone_map_sample.declare(m.float2)
```

The `fuse` function allows Slangpy to enter a "fusion context," where the call operator constructs a computational graph rather than immediately invoking a kernel.

The use of `vcast` in earlier examples was to ensure type constructors are available for use in fused calls:

```python
def tone_map_sample_func(u, v):
    return tone_map(sample_texture(float2(u, v)), 1, 0.5)

# Create a fused version of the function
tone_map_sample = fuse(tone_map_sample_func)

# Call it as usual
ubuffer = make_buffer_of_floats()
vbuffer = make_buffer_of_floats()
results = tone_map_sample(ubuffer, vbuffer)
```

