// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test shader for SlangComponentType API.

interface ITransform {
    float apply(float x);
}

struct ScaleBy2 : ITransform {
    float apply(float x) { return x * 2.0; }
}

struct AddOne : ITransform {
    float apply(float x) { return x + 1.0; }
}

float simple_add(float a, float b) {
    return a + b;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void compute_main(
    uint3 tid: SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> data,
    uniform uint count)
{
    uint idx = tid.x;
    if (idx >= count)
        return;
    data[idx] = data[idx] + 1.0;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void compute_generic<T : ITransform>(
    uint3 tid: SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> data,
    uniform uint count)
{
    uint idx = tid.x;
    if (idx >= count)
        return;
    T t;
    data[idx] = t.apply(data[idx]);
}
