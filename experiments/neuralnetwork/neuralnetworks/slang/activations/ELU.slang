// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off
implementing NeuralNetworks;

__include IScalarActivation;
__include basetypes.DiffCoopVec;

namespace Activation
{

public struct ELU<T : IReal, int Width> : IScalarActivation<T, Width>
{
    public T a;

    [BackwardDifferentiable, PreferRecompute]
    T activate(T x)
    {
        return a * (exp(-min(x, T(0.0f))) - T(1.0f)) + max(x, T(0.0f));
    }
};

public extension<T : IReal, int Width> ELU<T, Width> : IModel<DiffCoopVec<T, Width>, DiffCoopVec<T, Width>>
{
    [BackwardDifferentiable]
    public DiffCoopVec<T, Width> forward(DiffCoopVec<T, Width> x)
    {
        return a * (exp(max(-x, DiffCoopVec<T, Width>(0.0f))) - 1.0f) + max(x, DiffCoopVec<T, Width>(0.0f));
    }
}

}
