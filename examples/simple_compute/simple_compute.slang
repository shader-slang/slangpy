// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

struct Processor {
    StructuredBuffer<uint> a;
    StructuredBuffer<uint> b;
    RWStructuredBuffer<uint> c;
    void process(uint index) { c[index] = a[index] + b[index]; }
};

ParameterBlock<Processor> processor;

[shader("compute")]
[numthreads(32, 1, 1)]
void compute_main(uint3 tid: SV_DispatchThreadID)
{
    uint index = tid.x;
    if (index < 1024)
        processor.process(index);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void main2(
    uint3 tid: SV_DispatchThreadID,
    uniform uint N,
    uniform StructuredBuffer<uint> a,
    uniform StructuredBuffer<uint> b,
    uniform RWStructuredBuffer<uint> c
)
{
    uint index = tid.x;
    if (index < N)
        c[index] = a[index] + b[index];
}
