// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

struct NestedStruct
{
    float aa;
    int bb;
    RWStructuredBuffer<float> nc;
}

struct MyStruct
{
    NestedStruct a;
    uint b;
    ParameterBlock<NestedStruct> nest_param_block;
    RWStructuredBuffer<float> c;
};

ParameterBlock<MyStruct> input_struct;

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 tid: SV_DispatchThreadID)
{
    input_struct.c[0] =
        input_struct.a.aa +
        input_struct.a.bb +
        input_struct.b +
        input_struct.nest_param_block.aa +
        input_struct.nest_param_block.bb +
        input_struct.nest_param_block.nc[0];
}
