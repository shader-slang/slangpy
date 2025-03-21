// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off
implementing NeuralNetworks;

__include basetypes.IModel;

// Base interface for activations that operate on a per-element basis
interface IScalarActivation<T : IReal, int Width>
{
    [BackwardDifferentiable, PreferRecompute] T activate(T x);
}

// Extension:
// Any type that implements IScalarActivation can be turned into an IModel that accepts
// an array of Reals or a vector of Reals, by calling activate() on each element of the input
public extension<
    T : IReal,
    int Width,
    Act : IScalarActivation<T, Width>
> Act
: IModel<T[Width], T[Width]>
, IModel<vector<T, Width>, vector<T, Width>>
{
    [BackwardDifferentiable]
    public T[Width] forward(T x[Width])
    {
        T y[Width];
        [ForceUnroll]
        for (int i = 0; i < Width; ++i)
            y[i] = activate(x[i]);
        return y;
    }

    [BackwardDifferentiable]
    public vector<T, Width> forward(vector<T, Width> x)
    {
        vector<T, Width> y;
        [ForceUnroll]
        for (int i = 0; i < Width; ++i)
            y[i] = activate(x[i]);
        return y;
    }
}
