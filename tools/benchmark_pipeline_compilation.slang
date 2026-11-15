// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Runtime input prevents constant folding; link-time constants produce distinct kernels.
extern static const int VARIANT;
extern static const int ROUNDS;

float signal(float x)
{
    float4 state = float4(x, x * 0.7, x * 1.3, x * 0.2);
    [ForceUnroll]
    for (int i = 0; i < ROUNDS; ++i)
    {
        float phase = (i + 1) * 0.017 + VARIANT * 0.031;
        state = sin(state.yzwx * float4(1.1, 0.9, 1.2, 0.8) + phase)
              + cos(state.wxyz * 0.6 - phase) * 0.35;
        state += dot(state, float4(0.03, -0.02, 0.01, 0.04));
    }
    return dot(state, float4(0.2, 0.3, 0.1, 0.4));
}

float network(float x)
{
    float4 state = float4(x, x * x, x * 0.5, 1.0);
    [ForceUnroll]
    for (int i = 0; i < ROUNDS; ++i)
    {
        float bias = 0.01 * (VARIANT + 1) + 0.003 * i;
        float4 next = float4(
            dot(state, float4(0.7, -0.2, 0.1, 0.3)),
            dot(state, float4(-0.1, 0.8, 0.2, -0.3)),
            dot(state, float4(0.3, 0.1, 0.6, 0.2)),
            dot(state, float4(0.2, -0.3, 0.4, 0.5))) + bias;
        state = next / (1.0 + abs(next)) + state.wxyz * 0.1;
    }
    return dot(state, float4(0.2, 0.1, 0.4, 0.3));
}

float orbit(float x)
{
    float3 p = float3(x, 0.2, -0.3);
    float total = 0.0;
    [ForceUnroll]
    for (int i = 0; i < ROUNDS; ++i)
    {
        p = abs(p) / (0.7 + dot(p, p)) - float3(0.6, 0.5, 0.4);
        p = p.yzx + float3(0.002 * VARIANT, 0.001 * i, 0.005);
        total += exp(-dot(p, p)) / (i + 1.0);
    }
    return total;
}
