// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off

__exported import NeuralModules;
__exported import ArrayModule;
__exported import DiffCoopVec;

interface ICoopVecModule<T : IReal, int NumInputs, int NumOutputs> : IModule<DiffCoopVec<T, NumInputs>, DiffCoopVec<T, NumOutputs>>
{
    [BackwardDifferentiable]
    OutputType forward(InputType x);
}

extension<T : IReal, int NumInputs, int NumOutputs, Module : ICoopVecModule<T, NumInputs, NumOutputs>> Module
{
    static const CoopVecComponentType ComponentType = InputType.ComponentType;
}
extension<T : IReal, int NumInputs, int NumOutputs, Module : ICoopVecModule<T, NumInputs, NumOutputs>> Module : IArrayModule<T, NumInputs, NumOutputs>
{
    [BackwardDifferentiable]
    T[NumOutputs] forward(T[NumInputs] x)
    {
        return forward(DiffCoopVec<T, NumInputs>::fromArray(x)).toArray();
    }
}

struct CoopVecToArray<T : IReal, int N> : IModule<DiffCoopVec<T, N>, T[N]>
{
    [BackwardDifferentiable]
    OutputType forward(InputType x)
    {
        return x.toArray();
    }
}
struct ArrayToCoopVec<T : IReal, int N> : IModule<T[N], DiffCoopVec<T, N>>
{
    [BackwardDifferentiable]
    OutputType forward(InputType x)
    {
        return OutputType::fromArray(x);
    }
}
