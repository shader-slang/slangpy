# Nested fusion

This covers how we'll implement the calling of kernel functions using apparently complex nested parameters. This example shows a cut down version of a slang py generated kernel:

## Current unfused system

User function is written as a standard slang function:
```
user.slang

void user_func(float32_t a, float32_t b, out float32_t c) {
    //bla
}

```

Python call (in this case) creates some buffers and passes it in:

```
call.py

a = Buffer(bla)
b = Buffer(bla)
c = Buffer(bla)
user_func(a,b,c)
```

Generated code contains a kernel that unpacks, calls a trampoline function then packs up the results. This is already a very reduced form of fusion - the trampoline wrapping the user_func.

```
generated.slang (psuedo code)

void _trampoline( float32_t a,  float32_t b, out float32_t c) {
    user_func(a, b, c);
}

struct CallData {
    //buffers here
}

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    if (any(tid >= TOTAL_THREADS))
        return;

    //read in / inout values and define out values
    auto a = calldata.a_buffer.read(a_index(tid))
    auto b = calldata.b_buffer.read(b_index(tid))
    float c;

    //call trampoline function
    _trampoline(a,b,c);

    //write out / inout values
    calldata.c_buffer.write(c, c_index(tid))
}
```

## Extending to support nested parameters with a fusion approach

Given the following function, we want to support calls that generate the ray structure from any arbitrary combination of SOA/AOS style buffers:

```
void user_func(Ray ray, float max_dist, out float3 hit) {
    //bla
}
```

```
call.py

user_func(ray = { 
            'position': {
                'x' = 10,
                'y' = 20,
                'z' = z_buffer
            },
            'direction': dir_buffer },
          max_dist = 10,
          result = results_buffer)
```

When you analyse it, this is actually surprisingly simple. We already understand a flat set of function arguments. The user function now has to do slightly more complex type walking, but **only to cover the nested values**. This is critical, as it reduces the amount of work and understanding slangpy needs.

With this in mind, the process simply becomes:
- Walk the parameters alongside reflection, to identify a flat list of parameters that're being passed in, and what types/shapes they correspond to
- Generate a standard kernel/trampoline call that reads the flat parameter list
- Create a tree of fused calls that perfectly mimics the nested arguments

The result is some fairly trivial modifications to the existing kernels:

```
float3 fuse_0(float x, float y, float z) {
    return float3(x,y,z)
}
Ray fuse_1(float3 position, float3 direction) {
    return Ray(position, direction)
}

void _trampoline( float x, float y, float z, float3 direction, float max_distance, out float3 result) {
    user_func(fuse_1(fuse_0(x,y,z), direction), max_distance, result);
}
```

As a bonus, this can also drive how we think about a more full fusion system as a later milestone!