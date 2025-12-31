// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test shader for generic entrypoints.

interface IProcessor
{
    float process(float x);
}

struct DoubleIt : IProcessor
{
    float process(float x) { return x * 2.0; }
}

struct SquareIt : IProcessor
{
    float process(float x) { return x * x; }
}

struct AddTen : IProcessor
{
    float process(float x) { return x + 10.0; }
}

// Generic entrypoint that can be specialized with different IProcessor implementations.
[shader("compute")]
[numthreads(64, 1, 1)]
void compute_main<T : IProcessor>(
    uint3 tid: SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> data,
    uniform uint count)
{
    uint idx = tid.x;
    if (idx >= count)
        return;
    
    T processor;
    data[idx] = processor.process(data[idx]);
}

// Non-generic entrypoint for comparison.
[shader("compute")]
[numthreads(64, 1, 1)]
void compute_double(
    uint3 tid: SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> data,
    uniform uint count)
{
    uint idx = tid.x;
    if (idx >= count)
        return;
    
    data[idx] = data[idx] * 2.0;
}
