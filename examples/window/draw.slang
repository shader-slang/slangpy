// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

RWTexture2D g_output;
uniform uint g_frame;
uniform float2 g_mouse_pos;
uniform bool g_mouse_down;
uniform float g_time;
uniform float g_mouse_radius;
uniform float g_noise_scale;
uniform float g_noise_amount;

// based on https://github.com/keijiro/NoiseShader/blob/master/Assets/GLSL/SimplexNoise2D.glsl
// which itself is modification of https://github.com/ashima/webgl-noise/blob/master/src/noise3D.glsl
//
// License : Copyright (C) 2011 Ashima Arts. All rights reserved.
//           Distributed under the MIT License. See LICENSE file.
//           https://github.com/keijiro/NoiseShader/blob/master/LICENSE
//           https://github.com/ashima/webgl-noise
//           https://github.com/stegu/webgl-noise

float3 mod289(float3 x)
{
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

float2 mod289(float2 x)
{
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

float3 permute(float3 x)
{
    return mod289((x * 34.0 + 1.0) * x);
}

float3 taylorInvSqrt(float3 r)
{
    return 1.79284291400159 - 0.85373472095314 * r;
}

// output noise is in range [-1, 1]
float snoise(float2 v)
{
    const float4 C = float4(
        0.211324865405187,  // (3.0-sqrt(3.0))/6.0
        0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
        -0.577350269189626, // -1.0 + 2.0 * C.x
        0.024390243902439
    ); // 1.0 / 41.0

    // First corner
    float2 i = floor(v + dot(v, C.yy));
    float2 x0 = v - i + dot(i, C.xx);

    // Other corners
    float2 i1;
    i1.x = step(x0.y, x0.x);
    i1.y = 1.0 - i1.x;

    // x1 = x0 - i1  + 1.0 * C.xx;
    // x2 = x0 - 1.0 + 2.0 * C.xx;
    float2 x1 = x0 + C.xx - i1;
    float2 x2 = x0 + C.zz;

    // Permutations
    i = mod289(i); // Avoid truncation effects in permutation
    float3 p = permute(permute(i.y + float3(0.0, i1.y, 1.0)) + i.x + float3(0.0, i1.x, 1.0));

    float3 m = max(0.5 - float3(dot(x0, x0), dot(x1, x1), dot(x2, x2)), 0.0);
    m = m * m;
    m = m * m;

    // Gradients: 41 points uniformly over a line, mapped onto a diamond.
    // The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)
    float3 x = 2.0 * frac(p * C.www) - 1.0;
    float3 h = abs(x) - 0.5;
    float3 ox = floor(x + 0.5);
    float3 a0 = x - ox;

    // Normalise gradients implicitly by scaling m
    m *= taylorInvSqrt(a0 * a0 + h * h);

    // Compute final noise value at P
    float3 g = float3(a0.x * x0.x + h.x * x0.y, a0.y * x1.x + h.y * x1.y, a0.z * x2.x + h.z * x2.y);
    return 130.0 * dot(m, g);
}

float snoise01(float2 v)
{
    return snoise(v) * 0.5 + 0.5;
}

float2 repeat(float2 p, float size)
{
    return frac(p / size + 0.5) - 0.5;
}

float sdf_circle(float2 p, float r)
{
    return length(p) - r;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void compute_main(uint3 tid: SV_DispatchThreadID)
{
    uint2 dims;
    uint2 pixel = tid.xy;
    g_output.GetDimensions(dims.x, dims.y);
    if (any(pixel >= dims))
        return;

    float2 uv = float2(pixel) / dims.x;

    uv += g_time * 0.1;

    uv.x += snoise(uv * g_noise_scale) * g_noise_amount;
    uv.y += snoise((uv + float2(13, 17)) * g_noise_scale) * g_noise_amount;
    float2 p = repeat(uv, 0.2);
    float d = sdf_circle(p, 0.25);
    float3 color = float3(smoothstep(0.2, -0.2, d));

    color += smoothstep(g_mouse_radius, 0, distance(pixel, g_mouse_pos)) * float3(1, 0, 0);

    g_output[pixel] = float4(color, 1);
}
