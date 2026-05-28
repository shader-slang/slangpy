// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

struct Test {
    bool f_bool;
    bool2 f_bool2;
    bool3 f_bool3;
    bool4 f_bool4;
    int f_int;
    int2 f_int2;
    int3 f_int3;
    int4 f_int4;
    uint f_uint;
    uint2 f_uint2;
    uint3 f_uint3;
    uint4 f_uint4;
    float f_float;
    float2 f_float2;
    float3 f_float3;
    float4 f_float4;

    bool f_bool_array[1];
    int f_int_array[2];
    uint f_uint_array[3];
    float f_float_array[4];
};

// Global uniforms
uniform bool u_bool_false;
uniform bool u_bool_true;

uniform bool2 u_bool2;
uniform bool3 u_bool3;
uniform bool4 u_bool4;

uniform int u_int;
uniform int u_int_min;
uniform int u_int_max;

uniform int2 u_int2;
uniform int2 u_int2_min;
uniform int2 u_int2_max;

uniform int3 u_int3;
uniform int3 u_int3_min;
uniform int3 u_int3_max;

uniform int4 u_int4;
uniform int4 u_int4_min;
uniform int4 u_int4_max;

uniform uint u_uint;
uniform uint u_uint_min;
uniform uint u_uint_max;

uniform uint2 u_uint2;
uniform uint2 u_uint2_min;
uniform uint2 u_uint2_max;

uniform uint3 u_uint3;
uniform uint3 u_uint3_min;
uniform uint3 u_uint3_max;

uniform uint4 u_uint4;
uniform uint4 u_uint4_min;
uniform uint4 u_uint4_max;

uniform float u_float;
uniform float u_float_min;
uniform float u_float_max;

uniform float2 u_float2;
uniform float2 u_float2_min;
uniform float2 u_float2_max;

uniform float3 u_float3;
uniform float3 u_float3_min;
uniform float3 u_float3_max;

uniform float4 u_float4;
uniform float4 u_float4_min;
uniform float4 u_float4_max;

uniform float2x2 u_float2x2;
uniform float3x3 u_float3x3;
uniform float2x4 u_float2x4;
uniform float3x4 u_float3x4;
uniform float4x4 u_float4x4;

uniform float16_t u_float16_t;
uniform float16_t u_float16_t_min;
uniform float16_t u_float16_t_max;

uniform float16_t2 u_float16_t2;
uniform float16_t2 u_float16_t2_min;
uniform float16_t2 u_float16_t2_max;

uniform float16_t3 u_float16_t3;
uniform float16_t3 u_float16_t3_min;
uniform float16_t3 u_float16_t3_max;

uniform float16_t4 u_float16_t4;
uniform float16_t4 u_float16_t4_min;
uniform float16_t4 u_float16_t4_max;

uniform bool u_bool_array[4];
uniform int u_int_array[4];
uniform uint u_uint_array[4];
uniform float u_float_array[4];

uniform Test u_struct;
uniform int u_int_array_2[4];
uniform Test u_struct_array[4];

Buffer<uint> u_buffer;


RWStructuredBuffer<uint> results;

struct Writer {
    RWStructuredBuffer<uint> buffer;
    uint offset;

    __init(RWStructuredBuffer<uint> buffer)
    {
        this.buffer = buffer;
        this.offset = 0;
    }

    [mutating]
    void write<T : IWritable>(T value) { value.write(this); }

    [mutating]
    void write<T : IWritable, let N : int>(vector<T, N> value)
    {
        for (int i = 0; i < N; ++i)
            value[i].write(this);
    }

    [mutating]
    void write<T : IWritable, let R : int, let C : int>(matrix<T, R, C> value)
    {
        // Write row-major
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c)
                value[r][c].write(this);
    }
}

interface IWritable {
    void write(inout Writer writer);
}

extension bool : IWritable
{
    void write(inout Writer writer)
    {
        writer.buffer[writer.offset++] = asuint(this);
    }
}

extension int : IWritable
{
    void write(inout Writer writer)
    {
        writer.buffer[writer.offset++] = asuint(this);
    }
}

extension uint : IWritable
{
    void write(inout Writer writer)
    {
        writer.buffer[writer.offset++] = this;
    }
}

extension float : IWritable
{
    void write(inout Writer writer)
    {
        writer.buffer[writer.offset++] = asuint(this);
    }
}

extension float16_t : IWritable
{
    void write(inout Writer writer)
    {
        writer.buffer[writer.offset++] = f32tof16(this);
    }
}


[shader("compute")]
[numthreads(1, 1, 1)]
void compute_main(uint3 tid: SV_DispatchThreadID)
{
    if (any(tid != uint3(0)))
        return;

    Writer writer = Writer(results);

    writer.write(u_bool_false);
    writer.write(u_bool_true);

    writer.write(u_bool2);
    writer.write(u_bool3);
    writer.write(u_bool4);

    writer.write(u_int);
    writer.write(u_int_min);
    writer.write(u_int_max);

    writer.write(u_int2);
    writer.write(u_int2_min);
    writer.write(u_int2_max);

    writer.write(u_int3);
    writer.write(u_int3_min);
    writer.write(u_int3_max);

    writer.write(u_int4);
    writer.write(u_int4_min);
    writer.write(u_int4_max);

    writer.write(u_uint);
    writer.write(u_uint_min);
    writer.write(u_uint_max);

    writer.write(u_uint2);
    writer.write(u_uint2_min);
    writer.write(u_uint2_max);

    writer.write(u_uint3);
    writer.write(u_uint3_min);
    writer.write(u_uint3_max);

    writer.write(u_uint4);
    writer.write(u_uint4_min);
    writer.write(u_uint4_max);

    writer.write(u_float);
    writer.write(u_float_min);
    writer.write(u_float_max);

    writer.write(u_float2);
    writer.write(u_float2_min);
    writer.write(u_float2_max);

    writer.write(u_float3);
    writer.write(u_float3_min);
    writer.write(u_float3_max);

    writer.write(u_float4);
    writer.write(u_float4_min);
    writer.write(u_float4_max);

    writer.write(u_float2x2);
    writer.write(u_float3x3);
    writer.write(u_float2x4);
    writer.write(u_float3x4);
    writer.write(u_float4x4);

    writer.write(u_float16_t);
    writer.write(u_float16_t_min);
    writer.write(u_float16_t_max);

    writer.write(u_float16_t2);
    writer.write(u_float16_t2_min);
    writer.write(u_float16_t2_max);

    writer.write(u_float16_t3);
    writer.write(u_float16_t3_min);
    writer.write(u_float16_t3_max);

    writer.write(u_float16_t4);
    writer.write(u_float16_t4_min);
    writer.write(u_float16_t4_max);

    for (uint i = 0; i < 4; ++i)
        writer.write(u_bool_array[i]);
    for (uint i = 0; i < 4; ++i)
        writer.write(u_int_array[i]);
    for (uint i = 0; i < 4; ++i)
        writer.write(u_uint_array[i]);
    for (uint i = 0; i < 4; ++i)
        writer.write(u_float_array[i]);

    writer.write(u_struct.f_bool);
    writer.write(u_struct.f_bool2);
    writer.write(u_struct.f_bool3);
    writer.write(u_struct.f_bool4);
    writer.write(u_struct.f_int);
    writer.write(u_struct.f_int2);
    writer.write(u_struct.f_int3);
    writer.write(u_struct.f_int4);
    writer.write(u_struct.f_uint);
    writer.write(u_struct.f_uint2);
    writer.write(u_struct.f_uint3);
    writer.write(u_struct.f_uint4);
    writer.write(u_struct.f_float);
    writer.write(u_struct.f_float2);
    writer.write(u_struct.f_float3);
    writer.write(u_struct.f_float4);

    for (uint i = 0; i < 1; ++i)
        writer.write(u_struct.f_bool_array[i]);
    for (uint i = 0; i < 2; ++i)
        writer.write(u_struct.f_int_array[i]);
    for (uint i = 0; i < 3; ++i)
        writer.write(u_struct.f_uint_array[i]);
    for (uint i = 0; i < 4; ++i)
        writer.write(u_struct.f_float_array[i]);

    for (uint i = 0; i < 4; ++i)
        writer.write(u_int_array_2[i]);

    writer.write(u_struct_array[0].f_int);
    writer.write(u_struct_array[0].f_uint);
    writer.write(u_struct_array[0].f_float);
    writer.write(u_struct_array[1].f_int);
    writer.write(u_struct_array[1].f_uint);
    writer.write(u_struct_array[1].f_float);
    writer.write(u_struct_array[2].f_int);
    writer.write(u_struct_array[2].f_uint);
    writer.write(u_struct_array[2].f_float);
    writer.write(u_struct_array[3].f_int);
    writer.write(u_struct_array[3].f_uint);
    writer.write(u_struct_array[3].f_float);

    // writer.write(u_buffer);
}
