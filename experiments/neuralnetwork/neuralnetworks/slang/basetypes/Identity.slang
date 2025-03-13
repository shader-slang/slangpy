// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off

implementing NeuralNetworks;

__include IModel;

// Trivial model that just returns its input unmodified.
public struct Identity<T : IDifferentiable> : IModel<T, T>
{
    [BackwardDifferentiable]
    public IOType forward(IOType x)
    {
        return x;
    }
}
