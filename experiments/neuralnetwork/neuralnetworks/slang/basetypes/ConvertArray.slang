// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off
implementing NeuralNetworks;

__include IReal;
__include IModel;

[BackwardDifferentiable, ForceInline]
To convertReal<To : IReal, From : IReal>(From x)
{
    if (let z = x as To)
        return z;
    return To(x.toFloat());
}

public struct ConvertArrayPrecision<InElem : IReal, OutElem : IReal, int Width>
    : IModel<InElem[Width], OutElem[Width]>
    , IModel<vector<InElem, Width>, vector<OutElem, Width>>
    , IModel<DiffCoopVec<InElem, Width>, DiffCoopVec<OutElem, Width>>
{
    [BackwardDifferentiable]
    public OutElem[Width] forward(InElem x[Width])
    {
        OutElem[Width] y;
        [ForceUnroll]
        for (int i = 0; i < Width; ++i)
            y[i] = convertReal<OutElem>(x[i].toFloat());
        return y;
    }

    [BackwardDifferentiable]
    public vector<OutElem, Width> forward(vector<InElem, Width> x)
    {
        return vector<OutElem, Width>(x);
    }

    [BackwardDifferentiable]
    public DiffCoopVec<OutElem, Width> forward(DiffCoopVec<InElem, Width> x)
    {
        return DiffCoopVec<OutElem, Width>(x);
    }
}

public struct ConvertToArray<T : IReal, int Width>
    : IModel<vector<T, Width>, T[Width]>
    , IModel<DiffCoopVec<T, Width>, T[Width]>
{
    [BackwardDifferentiable]
    public T[Width] forward(vector<T, Width> x)
    {
        T[Width] y;
        [ForceUnroll]
        for (int i = 0; i < Width; ++i)
            y[i] = x[i];
        return y;
    }

    [BackwardDifferentiable]
    public T[Width] forward(DiffCoopVec<T, Width> x)
    {
        return x.toArray();
    }
}

public struct ConvertToVector<T : IReal, int Width>
    : IModel<T[Width], vector<T, Width>>
    , IModel<DiffCoopVec<T, Width>, vector<T, Width>>
{
    [BackwardDifferentiable]
    public vector<T, Width> forward(DiffCoopVec<T, Width> x)
    {
        return x.toVector();
    }

    [BackwardDifferentiable]
    public vector<T, Width> forward(T x[Width])
    {
        vector<T, Width> y;
        [ForceUnroll]
        for (int i = 0; i < Width; ++i)
            y[i] = x[i];
        return y;
    }
}

public struct ConvertToCoopVec<T : IReal, int Width> : IModel<T[Width], DiffCoopVec<T, Width>>, IModel<vector<T, Width>, DiffCoopVec<T, Width>>
{
    [BackwardDifferentiable]
    public DiffCoopVec<T, Width> forward(T x[Width])
    {
        return DiffCoopVec<T, Width>::fromArray(x);
    }

    [BackwardDifferentiable]
    public DiffCoopVec<T, Width> forward(vector<T, Width> x)
    {
        return DiffCoopVec<T, Width>::fromVector(x);
    }
}
