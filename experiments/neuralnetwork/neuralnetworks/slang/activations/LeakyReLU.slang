// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off
implementing NeuralNetworks;

__include IScalarActivation;
__include basetypes.DiffCoopVec;

namespace Activation
{

public struct LeakyReLU<T : IReal, int Width> : IScalarActivation<T, Width>
{
    public T negativeSlope;

    [BackwardDifferentiable, PreferRecompute]
    T activate(T x)
    {
        return max(x, T(0.0f)) + min(x, T(0.0f)) * negativeSlope;
    }
};

public extension<T : IReal, int Width> LeakyReLU<T, Width> : IModel<DiffCoopVec<T, Width>, DiffCoopVec<T, Width>>
{
    [BackwardDifferentiable]
    public DiffCoopVec<T, Width> forward(DiffCoopVec<T, Width> x)
    {
        return max(x, DiffCoopVec<T, Width>(0.0f)) - max(-x, DiffCoopVec<T, Width>(0.0f)) * negativeSlope;
    }
}

}
