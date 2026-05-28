// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

RWStructuredBuffer<uint> result;

struct S1 {
    float2 a;
    uint b;
};

struct S2 {
    bool3 a;
    S1 s1; // 3 values
    float b;
    uint2 c;
};

struct S3 {
    uint a;
    bool b;
    S2 s2; // 9 values
    float3 c;
    S1 s1; // 3 values
};

cbuffer data
{
    float a;
    S3 s3; // 17 values
    S2 s2; // 9 values
}

[shader("compute")]
[numthreads(1, 1, 1)]
void compute_main()
{
    result[0] = asuint(a);
    result[1] = s3.a;
    result[2] = s3.b ? 1 : 0;
    result[3] = s3.s2.a.x ? 1 : 0;
    result[4] = s3.s2.a.y ? 1 : 0;
    result[5] = s3.s2.a.z ? 1 : 0;
    result[6] = asuint(s3.s2.s1.a.x);
    result[7] = asuint(s3.s2.s1.a.y);
    result[8] = s3.s2.s1.b;
    result[9] = asuint(s3.s2.b);
    result[10] = s3.s2.c.x;
    result[11] = s3.s2.c.y;
    result[12] = asuint(s3.c.x);
    result[13] = asuint(s3.c.y);
    result[14] = asuint(s3.c.z);
    result[15] = asuint(s3.s1.a.x);
    result[16] = asuint(s3.s1.a.y);
    result[17] = s3.s1.b;
    result[18] = s2.a.x ? 1 : 0;
    result[19] = s2.a.y ? 1 : 0;
    result[20] = s2.a.z ? 1 : 0;
    result[21] = asuint(s2.s1.a.x);
    result[22] = asuint(s2.s1.a.y);
    result[23] = s2.s1.b;
    result[24] = asuint(s2.b);
    result[25] = s2.c.x;
    result[26] = s2.c.y;
}
