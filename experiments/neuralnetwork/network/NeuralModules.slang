// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off

typealias IScalar = __BuiltinFloatingPointType;

interface IModule<T : IScalar, int NumInputs, int NumOutputs>
{
    [BackwardDifferentiable]
    T[NumOutputs] forward(T x[NumInputs]);
}

// Chain two modules together, i.e. pass the output of the first to the second
// Can be nested arbitrarily
struct ModuleChain<
    T : IScalar,
    int NumInputs, int NumMiddle, int NumOutputs,
    First : IModule<T, NumInputs, NumMiddle>,
    Second : IModule<T, NumMiddle, NumOutputs>
> : IModule<T, NumInputs, NumOutputs>
{
    First first;
    Second second;

    [BackwardDifferentiable]
    T[NumOutputs] forward(T x[NumInputs])
    {
        let middle = first.forward(x);
        return second.forward(middle);
    }
}

// Extensions: An array IModule can also be used as a vector IModule
extension<T : IScalar, int NumInputs, int NumOutputs, Module : IModule<T, NumInputs, NumOutputs>> Module
{
    [BackwardDifferentiable]
    vector<T, NumOutputs> forward(vector<T, NumInputs> x)
    {
        T xArray[NumInputs];
        [ForceUnroll]
        for (int i = 0; i < NumInputs; ++i)
            xArray[i] = x[i];

        let yArray = forward(xArray);

        vector<T, NumOutputs> y;
        [ForceUnroll]
        for (int i = 0; i < NumOutputs; ++i)
            y[i] = yArray[i];

        return y;
    }
}
