// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
struct NthRootLossMapping
{
    float n;

    __init(float root) { this.n = root; }

    [BackwardDifferentiable]
    float3 eval(float3 x) { return n * (pow(x, 1.f / n) - 1.f); }
}

struct L1Loss
{
    __init() {}

    [BackwardDifferentiable]
    float eval(float3 pred, float3 ref)
    {
        float3 diff = abs(ref - pred);
        return (diff[0] + diff[1] + diff[2]) / 3.f;
    }
}

struct L2Loss
{
    __init() {}

    [BackwardDifferentiable]
    float eval(float3 pred, float3 ref)
    {
        float3 diff = (pred - ref);
        diff *= diff;
        return (diff[0] + diff[1] + diff[2]) / 3.f;
    }
}
