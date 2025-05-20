// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

[__AttributeUsage(_AttributeTargets.Var)]
struct rootAttribute { };

// ByteAddressBuffer

[shader("compute")]
[numthreads(32, 1, 1)]
void copy_byte_address_buffer(
    uint3 tid: SV_DispatchThreadID,
    ByteAddressBuffer src,
    RWByteAddressBuffer dst,
    uniform uint src_offset,
    uniform uint dst_offset,
    uniform uint count
)
{
    uint i = tid.x;
    if (i < count) {
        uint value = src.Load((src_offset + i) * 4);
        dst.Store((dst_offset + i) * 4, value);
    }
}

// Buffer<uint>

[shader("compute")]
[numthreads(32, 1, 1)]
void copy_buffer_uint(
    uint3 tid: SV_DispatchThreadID,
    Buffer<uint> src,
    RWBuffer<uint> dst,
    uniform uint src_offset,
    uniform uint dst_offset,
    uniform uint count
)
{
    uint i = tid.x;
    if (i < count)
        dst[dst_offset + i] = src[src_offset + i];
}

// StructuredBuffer<uint>

[shader("compute")]
[numthreads(32, 1, 1)]
void copy_structured_buffer_uint(
    uint3 tid: SV_DispatchThreadID,
    StructuredBuffer<uint> src,
    RWStructuredBuffer<uint> dst,
    uniform uint src_offset,
    uniform uint dst_offset,
    uniform uint count
)
{
    uint i = tid.x;
    if (i < count)
        dst[dst_offset + i] = src[src_offset + i];
}
