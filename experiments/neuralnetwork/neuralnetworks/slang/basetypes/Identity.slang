// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off

implementing NeuralNetworks;

__include IModel;

public struct Identity<T : IDifferentiable> : IModel<T, T>
{
    [BackwardDifferentiable]
    public IOType forward(IOType x)
    {
        return x;
    }
}
