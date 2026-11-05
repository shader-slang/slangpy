// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

__exported import bufferptr;

// Regular CI binding and scalar-access cases.

uint read_value(BufferPtr<> src, uint index)
{
    return src.as<uint>()[index];
}

uint read_advanced(BufferPtr<> src)
{
    let p = src.byte_offset(8).with_load_alignment<4>().as<uint>();
    return p[0] + p.element_offset(1)[0];
}

uint write_value(RWBufferPtr<> dst, uint value)
{
    dst.as<uint>()[0] = value;
    return dst.as<uint>()[0];
}

struct PointerRecord
{
    uint tag;
    BufferPtr<> ptr;
    uint tail;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void read_record(StructuredBuffer<PointerRecord> records, RWStructuredBuffer<uint> result)
{
    result[0] = records[0].tag;
    result[1] = records[0].ptr.as<uint>()[1];
    result[2] = records[0].tail;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void write_table(StructuredBuffer<RWBufferPtr<>> pointers)
{
    pointers[0].as<uint>()[1] = 99;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void nulls(
    BufferPtr<> src,
    RWBufferPtr<> dst,
    StructuredBuffer<BufferPtr<>> pointer_table,
    RWStructuredBuffer<uint> result
)
{
    BufferPtr<> local = {};
    RWBufferPtr<> writable = {};
    result[0] = src.is_valid();
    result[1] = dst.is_valid();
    result[2] = local.is_valid();
    result[3] = writable.is_valid();
    result[4] = pointer_table[0].is_valid();
    result[5] = pointer_table[1].is_valid();
}
