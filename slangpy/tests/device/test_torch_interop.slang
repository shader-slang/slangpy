// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

StructuredBuffer<float> a;
StructuredBuffer<float> b;
RWStructuredBuffer<float> c;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    c[tid.x] = a[tid.x] + b[tid.x];
}
