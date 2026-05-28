// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

[shader("compute")]
[numthreads(32, 1, 1)]
void test_buffer_add_f16(uint tid: SV_DispatchThreadID, StructuredBuffer<float16_t> data, RWByteAddressBuffer result)
{
    float16_t value1 = data[tid];
    float16_t value2 = -data[tid];
    float16_t orig;
    result.InterlockedAddF16(0, value1, orig);
    result.InterlockedAddF16(2, value2, orig);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void test_buffer_add_f16x2(uint tid: SV_DispatchThreadID, StructuredBuffer<float16_t2> data, RWByteAddressBuffer result)
{
    uint value1 = reinterpret<uint>(data[tid]);
    uint value2 = reinterpret<uint>(-data[tid]);
#ifdef __TARGET_D3D12__
    result._NvInterlockedAddFp16x2(0, value1);
    result._NvInterlockedAddFp16x2(4, value2);
#endif
}

[shader("compute")]
[numthreads(32, 1, 1)]
void test_buffer_add_f32(uint tid: SV_DispatchThreadID, StructuredBuffer<float> data, RWByteAddressBuffer result)
{
    float value1 = data[tid];
    float value2 = -data[tid];
    result.InterlockedAddF32(0, value1);
    result.InterlockedAddF32(4, value2);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void test_buffer_add_u64(uint tid: SV_DispatchThreadID, StructuredBuffer<uint64_t> data, RWByteAddressBuffer result)
{
    uint64_t value1 = data[tid];
    uint64_t value2 = 0xffffffffffffffffull - data[tid];
    uint64_t orig;
    result.InterlockedAdd64(0, value1, orig);
    result.InterlockedAdd64(8, value2, orig);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void test_buffer_add_i64(uint tid: SV_DispatchThreadID, StructuredBuffer<int64_t> data, RWByteAddressBuffer result)
{
    int64_t value1 = data[tid];
    int64_t value2 = -data[tid];
    int64_t orig;
    result.InterlockedAddI64(0, value1, orig);
    result.InterlockedAddI64(8, value2, orig);
}

#define r32f [format("r32f")]

[shader("compute")]
[numthreads(32, 1, 1)]
void test_texture_add_f32_1d(
    uint tid: SV_DispatchThreadID,
    StructuredBuffer<float> data,
    r32f RWTexture1D<float> result
)
{
    float value1 = data[tid];
    float value2 = -data[tid];
    result.InterlockedAddF32(0, value1);
    result.InterlockedAddF32(1, value2);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void test_texture_add_f32_2d(
    uint tid: SV_DispatchThreadID,
    StructuredBuffer<float> data,
    r32f RWTexture2D<float> result
)
{
    float value1 = data[tid];
    float value2 = -data[tid];
    result.InterlockedAddF32(uint2(0), value1);
    result.InterlockedAddF32(uint2(1), value2);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void test_texture_add_f32_3d(
    uint tid: SV_DispatchThreadID,
    StructuredBuffer<float> data,
    r32f RWTexture3D<float> result
)
{
    float value1 = data[tid];
    float value2 = -data[tid];
    result.InterlockedAddF32(uint3(0), value1);
    result.InterlockedAddF32(uint3(1), value2);
}
