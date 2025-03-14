// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
implementing NeuralNetworks;

__include IModel;

// Chain two models together, i.e. pass the output of the first to the second
// Can be nested arbitrarily to chain as many models as needed
// E.g. ChainedModelPair<A, ChainedModelPair<B, C>> will execute models A, B, C
// in sequence.
public struct ChainedModelPair<
    InputT : IDifferentiable, MiddleT : IDifferentiable, OutputT : IDifferentiable,
    First : IModel<InputT, MiddleT>,
    Second : IModel<MiddleT, OutputT>,
> : IModel<InputT, OutputT>
{
    public First first;
    public Second second;

    [BackwardDifferentiable]
    public OutputT forward(InputT x)
    {
        MiddleT middle = first.forward(x);
        return second.forward(middle);
    }
}
