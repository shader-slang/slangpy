// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

__exported import bufferptr;

// Byte access and vector/matrix strides.

void read_interior<Access access>(BufferPtr<void, access> p, RWStructuredBuffer<uint4> result, uint result_index)
{
    result[result_index + 0] = p.load<uint4>();
    result[result_index + 1] = p.load<uint4>(16);
    result[result_index + 2] = p.load_aligned<uint4, 16>();
    result[result_index + 3] = p.load_aligned<uint4, 16>(16);
    result[result_index + 4] = p.with_load_alignment<16>().load<uint4>(16);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void interior_access(BufferPtr<> src, RWBufferPtr<> dst, RWStructuredBuffer<uint4> result)
{
    read_interior(src, result, 0);
    read_interior(dst, result, 8);
    result[5] = src.as<uint4>()[1];
    result[6] = src.with_load_alignment<16>().as<uint4>()[1];
    var advanced_src = src.as<uint4>();
    ++advanced_src;
    result[7] = advanced_src[0];
    result[13] = dst.as<uint4>()[1];
    result[14] = dst.with_load_alignment<16>().as<uint4>()[1];
    var advanced_dst = dst.as<uint4>();
    ++advanced_dst;
    result[15] = advanced_dst[0];
    let a = src.load<uint4>();
    let b = src.load<uint4>(16);
    dst.store<uint4>(a);
    dst.store<uint4>(16, b);
    dst.store_aligned<uint4, 16>(32, a + 100);
    dst.store_aligned<uint4, 16>(48, b + 200);
    dst.with_load_alignment<16>().store<uint4>(64, a + 300);
    dst.as<uint4>()[5] = b + 400;
    dst.with_load_alignment<16>().as<uint4>()[6] = a + 500;
    var advanced = dst.as<uint4>() + 6;
    ++advanced;
    advanced[0] = b + 600;
}


[shader("compute")]
[numthreads(1, 1, 1)]
void stride_vector3(
    BufferPtr<> src,
    RWBufferPtr<> dst,
    StructuredBuffer<float3> structured,
    RWStructuredBuffer<uint64_t> result
)
{
    var p = src.as<float3>();
    let indexed = p[1];
    ++p;
    let advanced = p[0];
    let output = dst.as<float3>();
    output[0] = indexed;
    output[1] = advanced;
#if defined(__TARGET_VULKAN__) || defined(__TARGET_CUDA__) || defined(__TARGET_METAL__)
    var native = Ptr<float3>(src.bits());
    ++native;
    let reference = native[0];
    result[0] = uint64_t(native) - src.bits();
#else
    let reference = structured[1];
    result[0] = (p.bits() >> 32) - (src.bits() >> 32);
#endif
    output[2] = reference;
    let buffer_value = structured[1];
    result[1] = uint(1.0 * indexed[0] + 2.0 * indexed[1] + 3.0 * indexed[2]);
    result[2] = uint(1.0 * advanced[0] + 2.0 * advanced[1] + 3.0 * advanced[2]);
    result[3] = uint(1.0 * reference[0] + 2.0 * reference[1] + 3.0 * reference[2]);
    result[4] = uint(1.0 * buffer_value[0] + 2.0 * buffer_value[1] + 3.0 * buffer_value[2]);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void stride_matrix3x4(
    BufferPtr<> src,
    RWBufferPtr<> dst,
    StructuredBuffer<float3x4> structured,
    RWStructuredBuffer<uint64_t> result
)
{
    var p = src.as<float3x4>();
    let indexed = p[1];
    ++p;
    let advanced = p[0];
    let output = dst.as<float3x4>();
    output[0] = indexed;
    output[1] = advanced;
#if defined(__TARGET_VULKAN__) || defined(__TARGET_CUDA__) || defined(__TARGET_METAL__)
    var native = Ptr<float3x4>(src.bits());
    ++native;
    let reference = native[0];
    result[0] = uint64_t(native) - src.bits();
#else
    let reference = structured[1];
    result[0] = (p.bits() >> 32) - (src.bits() >> 32);
#endif
    output[2] = reference;
    let buffer_value = structured[1];
    result[1] = uint(
        1.0 * indexed[0][0] + 2.0 * indexed[0][1] + 3.0 * indexed[0][2] + 4.0 * indexed[0][3] + 5.0 * indexed[1][0]
        + 6.0 * indexed[1][1] + 7.0 * indexed[1][2] + 8.0 * indexed[1][3] + 9.0 * indexed[2][0] + 10.0 * indexed[2][1]
        + 11.0 * indexed[2][2] + 12.0 * indexed[2][3]
    );
    result[2] = uint(
        1.0 * advanced[0][0] + 2.0 * advanced[0][1] + 3.0 * advanced[0][2] + 4.0 * advanced[0][3] + 5.0 * advanced[1][0]
        + 6.0 * advanced[1][1] + 7.0 * advanced[1][2] + 8.0 * advanced[1][3] + 9.0 * advanced[2][0]
        + 10.0 * advanced[2][1] + 11.0 * advanced[2][2] + 12.0 * advanced[2][3]
    );
    result[3] = uint(
        1.0 * reference[0][0] + 2.0 * reference[0][1] + 3.0 * reference[0][2] + 4.0 * reference[0][3]
        + 5.0 * reference[1][0] + 6.0 * reference[1][1] + 7.0 * reference[1][2] + 8.0 * reference[1][3]
        + 9.0 * reference[2][0] + 10.0 * reference[2][1] + 11.0 * reference[2][2] + 12.0 * reference[2][3]
    );
    result[4] = uint(
        1.0 * buffer_value[0][0] + 2.0 * buffer_value[0][1] + 3.0 * buffer_value[0][2] + 4.0 * buffer_value[0][3]
        + 5.0 * buffer_value[1][0] + 6.0 * buffer_value[1][1] + 7.0 * buffer_value[1][2] + 8.0 * buffer_value[1][3]
        + 9.0 * buffer_value[2][0] + 10.0 * buffer_value[2][1] + 11.0 * buffer_value[2][2] + 12.0 * buffer_value[2][3]
    );
}

[shader("compute")]
[numthreads(1, 1, 1)]
void stride_matrix4x3(
    BufferPtr<> src,
    RWBufferPtr<> dst,
    StructuredBuffer<float4x3> structured,
    RWStructuredBuffer<uint64_t> result
)
{
    var p = src.as<float4x3>();
    let indexed = p[1];
    ++p;
    let advanced = p[0];
    let output = dst.as<float4x3>();
    output[0] = indexed;
    output[1] = advanced;
#if defined(__TARGET_VULKAN__) || defined(__TARGET_CUDA__) || defined(__TARGET_METAL__)
    var native = Ptr<float4x3>(src.bits());
    ++native;
    let reference = native[0];
    result[0] = uint64_t(native) - src.bits();
#else
    let reference = structured[1];
    result[0] = (p.bits() >> 32) - (src.bits() >> 32);
#endif
    output[2] = reference;
    let buffer_value = structured[1];
    result[1] = uint(
        1.0 * indexed[0][0] + 2.0 * indexed[0][1] + 3.0 * indexed[0][2] + 4.0 * indexed[1][0] + 5.0 * indexed[1][1]
        + 6.0 * indexed[1][2] + 7.0 * indexed[2][0] + 8.0 * indexed[2][1] + 9.0 * indexed[2][2] + 10.0 * indexed[3][0]
        + 11.0 * indexed[3][1] + 12.0 * indexed[3][2]
    );
    result[2] = uint(
        1.0 * advanced[0][0] + 2.0 * advanced[0][1] + 3.0 * advanced[0][2] + 4.0 * advanced[1][0] + 5.0 * advanced[1][1]
        + 6.0 * advanced[1][2] + 7.0 * advanced[2][0] + 8.0 * advanced[2][1] + 9.0 * advanced[2][2]
        + 10.0 * advanced[3][0] + 11.0 * advanced[3][1] + 12.0 * advanced[3][2]
    );
    result[3] = uint(
        1.0 * reference[0][0] + 2.0 * reference[0][1] + 3.0 * reference[0][2] + 4.0 * reference[1][0]
        + 5.0 * reference[1][1] + 6.0 * reference[1][2] + 7.0 * reference[2][0] + 8.0 * reference[2][1]
        + 9.0 * reference[2][2] + 10.0 * reference[3][0] + 11.0 * reference[3][1] + 12.0 * reference[3][2]
    );
    result[4] = uint(
        1.0 * buffer_value[0][0] + 2.0 * buffer_value[0][1] + 3.0 * buffer_value[0][2] + 4.0 * buffer_value[1][0]
        + 5.0 * buffer_value[1][1] + 6.0 * buffer_value[1][2] + 7.0 * buffer_value[2][0] + 8.0 * buffer_value[2][1]
        + 9.0 * buffer_value[2][2] + 10.0 * buffer_value[3][0] + 11.0 * buffer_value[3][1] + 12.0 * buffer_value[3][2]
    );
}


// Pointer operations and alignment.
[shader("compute")]
[numthreads(1, 1, 1)]
void pointer_operations(BufferPtr<> src, RWBufferPtr<> dst, RWStructuredBuffer<uint64_t> result)
{
    let base = src.as<uint>();
    var p = base + 2;
    result[0] = p[-1];
    result[1] = p[int64_t(-2)];
    result[2] = p[uint64_t(1)];
    result[3] = (p - 1)[0];
    result[4] = (p++)[0];
    result[5] = p[0];
    result[6] = (--p)[0];
    result[7] = (p--)[0];
    result[8] = (++p)[0];
    result[9] = p == base + 2 && p != base && p > base && p >= base && base < p && base <= p;
    BufferPtr<> erased = BufferPtr<>(p);
    let restored = (BufferPtr<uint>)erased;
    result[10] = restored[0];
    result[11] = uint64_t(restored) == p.bits() && int64_t(restored) == int64_t(p.bits());
    BufferPtr<uint> empty = nullptr;
    result[12] = empty == nullptr && nullptr == empty && p != nullptr && nullptr != p && !empty;
    if (p)
        result[13] = 1;
    let writable = dst.as<uint>();
    writable[int64_t(1)] = p[-1];
    storeAligned<4>(writable, loadAligned<4>(p));
    result[14] = writable[1] + writable[0];

    // Inspect encoding arithmetic only: these synthetic addresses are never dereferenced.
#if defined(__TARGET_D3D12__)
    let far = BufferPtr<uint>(uint64_t(7) | (uint64_t(0xfffffffc) << 32));
    result[15] = (far - uint64_t(0x3fffffff)).bits() == 7;
    result[16] = (far + int64_t(-0x3fffffff)).bits() == 7;
#else
    let far = BufferPtr<uint>(uint64_t(0x1000000000));
    result[15] = (far + uint64_t(0x100000001)).bits() == uint64_t(0x1400000004);
    result[16] = (far + int64_t(-0x100000001)).bits() == uint64_t(0xbfffffffc);
#endif
    let wide = RWBufferPtr<uint64_t>(dst.bits());
    wide[2] = uint64_t(base[3]) + uint64_t(2);
    var next = wide;
    ++next;
    result[17] = uint(next[1]);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void pointer_alignment(BufferPtr<> src, RWStructuredBuffer<uint> result)
{
    var packed = src.with_load_alignment<16>().as<float3>().with_load_alignment<4>();
    result[0] = asuint(packed[1].x);
    ++packed;
    result[1] = asuint(packed[0].x);
    result[2] = asuint((packed - 1)[1].x);
    var pairs = src.with_load_alignment<8>().as<uint2>();
    result[3] = pairs[1].x;
    ++pairs;
    result[4] = pairs[0].x;
    // A load operand is preserved even when address movement would invalidate it.
    // Do not dereference this deliberately over-specified pointer.
    var unchecked = src.with_load_alignment<16>().as<float3>();
    ++unchecked;
    BufferPtr<float3, Access.Read, 16> retained = unchecked.byte_offset(-12);
    result[5] = retained.bits() == src.bits();
    let retuned = BufferPtr<float3, Access.Read, 16>(src.as<float3>());
    result[6] = retuned.bits() == src.bits();
}

[shader("compute")]
[numthreads(1, 1, 1)]
void copy_values(BufferPtr<> src, RWBufferPtr<> dst)
{
    dst.store_aligned<float4, 16>(16, src.load_aligned<float4, 16>(16));
    dst.store_aligned<uint64_t, 8>(64, src.load_aligned<uint64_t, 8>(64));
    dst.as<float3>()[3] = src.as<float3>()[3];
    dst.store<uint>(52, src.load<uint>(52));
    dst.byte_offset(56).store_aligned<uint, 4>(0, src.byte_offset(56).load_aligned<uint, 4>());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void plain_values(BufferPtr<> src, RWBufferPtr<> dst)
{
    dst.as<float4>()[1] = src.as<float4>()[1];
    dst.as<uint64_t>()[8] = src.as<uint64_t>()[8];
    dst.as<float3>()[3] = src.as<float3>()[3];
    dst.store<uint>(52, src.load<uint>(52));
    // Explicitly return to plain access after selecting an aligned view.
    dst.with_load_alignment<16>().with_load_alignment<0>().store<uint>(56, src.load<uint>(56));
}

// D3D plain vector loads have a four-byte floor; double stores still need eight.
// Native CUDA float4 pointers need more alignment, so this offset test is D3D-only.
#if defined(__TARGET_D3D12__)
[shader("compute")]
[numthreads(1, 1, 1)]
void plain_d3d_offset(BufferPtr<> src, RWBufferPtr<> dst)
{
    dst.store<float4>(4, src.load<float4>(4));
    dst.store<double>(24, src.load<double>(24));
}
#endif

[shader("compute")]
[numthreads(1, 1, 1)]
void update_values(RWBufferPtr<> dst)
{
    let values = dst.as<float3>();
    values[3] = values[3] + float3(1.0, 2.0, 3.0);
}

// Record layout, pointer tables and dynamic dispatch.
struct PackedRecord
{
    float3 position;
    uint tag;
    static const uint UNUSED = 19;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void aligned_records(BufferPtr<> src, RWBufferPtr<> dst)
{
    let input = src.as<PackedRecord>().with_load_alignment<16>();
    let output = dst.as<PackedRecord>().with_load_alignment<16>();
    output[1] = input[1];
    dst.as<void>().with_load_alignment<8>().store_aligned<uint64_t, 8>(32, src.load_aligned<uint64_t, 8>(32));
}

interface IPointerRecord
{
    uint evaluate();
}

struct DynamicPointerRecord : IPointerRecord
{
    uint tag;
    BufferPtr<> ptr;
    uint tail;
    uint evaluate() { return tag + ptr.as<uint>()[1] + tail; }
}

struct DynamicStorage
{
    uint4 words[8];
}

[shader("compute")]
[numthreads(1, 1, 1)]
void read_dynamic(StructuredBuffer<DynamicStorage> records, RWStructuredBuffer<uint> result)
{
    let value = createDynamicObject<IPointerRecord, DynamicStorage>(0, records[0]);
    result[0] = value.evaluate();
}

[shader("compute")]
[numthreads(32, 1, 1)]
void divergent(
    uint3 tid: SV_DispatchThreadID,
    uint count,
    StructuredBuffer<BufferPtr<>> pointer_table,
    RWStructuredBuffer<uint> result
)
{
    if (tid.x < count)
        result[tid.x] = pointer_table[tid.x % 8].as<uint>()[tid.x % 4];
}

// Padded aggregates, linked traversal and nested pointer marshalling.
struct PaddedInner
{
    uint64_t key;
    uint value;
}

struct NestedValues
{
    PaddedInner items[2];
    float3 positions[2];
    uint tail;
}

[shader("compute")]
[numthreads(4, 1, 1)]
void nested_values(
    uint3 tid: SV_DispatchThreadID,
    uniform BufferPtr<> src,
    uniform RWBufferPtr<> dst,
    RWStructuredBuffer<uint> result
)
{
    uint i = tid.x % 2;
    let p = src.as<NestedValues>();
    var advanced = p;
    ++advanced;
    result[tid.x] = uint(p[0].items[i].key) + p[0].items[i].value + uint(p[0].positions[i].z);
    result[4 + tid.x] = uint(p[1].items[i].key) + uint(advanced[0].positions[i].y) + p[1].tail;
    if (tid.x == 0) {
        dst.as<NestedValues>()[0] = p[1];
    }
}

struct UnpaddedNode
{
    BufferPtr<> next;
    BufferPtr<> payload;
    uint tag;
}

// Exercise implicit tail padding in actual pointer traversal as well as address arithmetic.
struct ChainNode
{
    BufferPtr<> next;
    BufferPtr<> payload;
    uint tag;
}

uint64_t pointer_distance(BufferPtr<> a, BufferPtr<> b)
{
#if defined(__TARGET_D3D12__)
    return (b.bits() >> 32) - (a.bits() >> 32);
#else
    return b.bits() - a.bits();
#endif
}

[shader("compute")]
[numthreads(1, 1, 1)]
void stride_metadata(uniform BufferPtr<> root, RWStructuredBuffer<uint64_t> result)
{
    result[0] = pointer_distance(root, root.as<PaddedInner>().element_offset(1).as<void>());
    result[1] = pointer_distance(root, root.as<UnpaddedNode>().element_offset(1).as<void>());
    result[2] = pointer_distance(root, root.as<NestedValues>().element_offset(1).as<void>());
#if defined(__TARGET_VULKAN__) || defined(__TARGET_CUDA__) || defined(__TARGET_METAL__)
    result[3] = uint64_t(Ptr<PaddedInner>(root.bits()) + 1) - root.bits();
    result[4] = uint64_t(Ptr<UnpaddedNode>(root.bits()) + 1) - root.bits();
    result[5] = uint64_t(Ptr<NestedValues>(root.bits()) + 1) - root.bits();
#else
    result[3] = 0;
    result[4] = 0;
    result[5] = 0;
#endif
    result[6] = pointer_distance(root, root.as<ChainNode>().element_offset(1).as<void>());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void build_chain(
    uniform BufferPtr<> nodes_ro,
    uniform RWBufferPtr<> nodes_rw,
    uniform BufferPtr<> a,
    uniform BufferPtr<> b
)
{
    var nodes = nodes_rw.as<ChainNode>();
    nodes[0] = { nodes_ro.as<ChainNode>().element_offset(1).as<void>(), a, 10 };
    nodes[1] = { nodes_ro.as<ChainNode>().element_offset(2).as<void>(), b, 20 };
    nodes[2] = { nullptr, a.byte_offset(4), 30 };
}

BufferPtr<ChainNode> select_chain(BufferPtr<> root, uint lane)
{
    BufferPtr<ChainNode> selected = nullptr;
    if (lane % 3 != 2)
        selected = root.as<ChainNode>() + (lane % 3);
    return selected;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void follow_chain(uint3 tid: SV_DispatchThreadID, uniform BufferPtr<> root, RWStructuredBuffer<uint2> result)
{
    if (tid.x >= 65)
        return;
    var p = select_chain(root, tid.x);
    uint sum = 0;
    uint count = 0;
    while (p && count < 4) {
        let node = p[0];
        sum += node.tag + node.payload.as<uint>()[0];
        p = node.next.as<ChainNode>();
        ++count;
    }
    result[tid.x] = uint2(sum, count);
}

struct InnerPointers
{
    BufferPtr<> pointers[2];
}

interface INestedPointers
{
    uint evaluate();
}

struct NestedPointers : INestedPointers
{
    InnerPointers inner[2];
    uint tag;
    uint evaluate()
    {
        return inner[0].pointers[0].as<uint>()[0] + inner[0].pointers[1].as<uint>()[0]
            + inner[1].pointers[0].as<uint>()[0] + inner[1].pointers[1].as<uint>()[0] + tag;
    }
}

NestedPointers make_nested(BufferPtr<> a, BufferPtr<> b)
{
    NestedPointers record;
    record.inner[0].pointers[0] = a;
    record.inner[0].pointers[1] = b;
    record.inner[1].pointers[0] = a.byte_offset(4);
    record.inner[1].pointers[1] = b.byte_offset(4);
    record.tag = 17;
    return record;
}

uint read_pointer_array(BufferPtr<> pointers[2])
{
    return pointers[0].as<uint>()[0] + pointers[1].as<uint>()[0];
}

void change_nested(inout NestedPointers record, BufferPtr<> replacement)
{
    record.inner[1].pointers[0] = replacement;
    record.tag += 3;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void nested_marshalling(
    uniform BufferPtr<> a,
    uniform BufferPtr<> b,
    uniform RWBufferPtr<> storage,
    RWStructuredBuffer<uint> result
)
{
    var record = make_nested(a, b);
    result[0] = read_pointer_array(record.inner[0].pointers);
    result[1] = record.evaluate();
    change_nested(record, b);
    result[2] = record.evaluate();
    storage.as<NestedPointers>()[0] = record;
    result[3] = storage.as<NestedPointers>() [0].evaluate();
}


[shader("compute")]
[numthreads(1, 1, 1)]
void nested_dynamic(
    StructuredBuffer<DynamicStorage> storage,
    uniform uint record_index,
    RWStructuredBuffer<uint> result
)
{
    let record = createDynamicObject<INestedPointers, DynamicStorage>(0, storage[record_index]);
    result[0] = record.evaluate();
}

// Sequential field updates: compare BufferPtr, structured buffers and native pointers.
struct AliasRecord
{
    uint x;
    uint y;
}

RWStructuredBuffer<AliasRecord> records;

void touch_structured(inout uint x)
{
    records[0].y = 99;
    x += 1;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void alias_structured()
{
    touch_structured(records[0].x);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void alias_structured_field()
{
    records[0].x = 7;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void alias_structured_compound()
{
    records[0].x += 7;
}

void touch_wrapper(inout uint x, RWBufferPtr<AliasRecord> p)
{
    p[0].y = 99;
    x += 1;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void alias_wrapper(uniform uint64_t bits)
{
    let p = RWBufferPtr<AliasRecord>(bits);
    touch_wrapper(p[0].x, p);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void alias_wrapper_field(uniform uint64_t bits)
{
    RWBufferPtr<AliasRecord>(bits)[0].x = 7;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void alias_wrapper_compound(uniform uint64_t bits)
{
    RWBufferPtr<AliasRecord>(bits)[0].x += 7;
}

#if defined(__TARGET_VULKAN__) || defined(__TARGET_CUDA__) || defined(__TARGET_METAL__)
void touch_native(inout uint x, Ptr<AliasRecord> p)
{
    p[0].y = 99;
    x += 1;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void alias_native(uniform uint64_t bits)
{
    let p = Ptr<AliasRecord>(bits);
    touch_native(p[0].x, p);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void alias_native_field(uniform uint64_t bits)
{
    Ptr<AliasRecord>(bits)[0].x = 7;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void alias_native_compound(uniform uint64_t bits)
{
    Ptr<AliasRecord>(bits)[0].x += 7;
}
#endif

// Additional Metal entry points also exercised by emitted-code checks.
#if defined(__TARGET_METAL__)
uint read_generic<Access access>(BufferPtr<uint, access> p)
{
    return p[0] + p[1];
}

[shader("compute")]
[numthreads(1, 1, 1)]
void typed_access(BufferPtr<> src, RWBufferPtr<> dst)
{
    dst.as<uint>() [1] = read_generic(src.as<uint>()) + read_generic(dst.as<uint>());
    dst.store<uint>(16, src.load<uint>(8));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void aligned_access(BufferPtr<> src, RWBufferPtr<> dst)
{
    dst.store_aligned<float4, 16>(16, src.load_aligned<float4, 16>(16));
    dst.with_load_alignment<16>().as<float4>()[2] = src.with_load_alignment<16>().as<float4>()[2];
}

[shader("compute")]
[numthreads(1, 1, 1)]
void vector_matrix_access(BufferPtr<> src, RWBufferPtr<> dst)
{
    var p = src.as<float3>();
    ++p;
    dst.as<float3>()[0] = p[0];
    dst.byte_offset(48).as<float3x4>()[1] = src.as<float3x4>()[1];
    dst.byte_offset(192).as<float4x3>()[1] = src.as<float4x3>()[1];
}

struct MetalPointerRecord
{
    uint tag;
    BufferPtr<> src;
    RWBufferPtr<> dst;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void pointer_tables(StructuredBuffer<MetalPointerRecord> table, uint index)
{
    let item = table[index];
    item.dst.as<uint>()[0] = item.src.as<uint>()[item.tag];
}

[shader("compute")]
[numthreads(1, 1, 1)]
void casts_and_null(RWBufferPtr<uint> p, RWStructuredBuffer<uint64_t> result)
{
    let read = BufferPtr<uint>(p);
    let restored = RWBufferPtr<uint>(read);
    BufferPtr<> null_ptr = nullptr;
    result[0] = uint64_t(read);
    result[1] = read == restored;
    result[2] = null_ptr == nullptr;
    result[3] = bool(null_ptr);
    result[4] = uint64_t(restored + 2);
}
#endif
