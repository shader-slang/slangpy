// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

struct MyStruct {
    int4 a;
    float4 b;
};

StructuredBuffer<uint> buffer_uint;
StructuredBuffer<float4> buffer_float4;
StructuredBuffer<MyStruct> buffer_my_struct;

[shader("compute")]
[numthreads(1, 1, 1)]
void compute_main() {}
