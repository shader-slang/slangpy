// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
implementing NeuralNetworks;

namespace Losses
{

public struct L2
{
    public __init() {}

    [BackwardDifferentiable]
    public float eval(float3 pred, float3 ref)
    {
        float3 diff = (pred - ref);
        diff *= diff;
        return (diff[0] + diff[1] + diff[2]) / 3.f;
    }
}

}
