// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sgl.device.print;

[shader("compute")]
[numthreads(1, 1, 1)]
void compute_main(uint3 tid: SV_DispatchThreadID)
{
    print("Hello from precompiled module!");
}

struct Vec3
{
    float x;
    float y;
    float z;
};

float add_floats(float a, float b)
{
    return a + b;
}

Vec3 create_vec3(float x, float y, float z)
{
    Vec3 v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}

float dot_product(Vec3 a, Vec3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
