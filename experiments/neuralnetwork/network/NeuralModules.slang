// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off

interface IModule<InputT : IDifferentiable, OutputT : IDifferentiable>
{
    [BackwardDifferentiable]
    OutputT forward(InputT x);
}

extension<InputT : IDifferentiable, OutputT : IDifferentiable, Module : IModule<InputT, OutputT>> Module
{
    typealias InputType = InputT;
    typealias OutputType = OutputT;
}

// Chain two modules together, i.e. pass the output of the first to the second
// Can be nested arbitrarily
struct ChainedModulePair<
    InputT : IDifferentiable, MiddleT : IDifferentiable, OutputT : IDifferentiable,
    First : IModule<InputT, MiddleT>,
    Second : IModule<MiddleT, OutputT>
> : IModule<InputT, OutputT>
{
    First first;
    Second second;

    [BackwardDifferentiable]
    OutputT forward(InputT x)
    {
        MiddleT middle = first.forward(x);
        return second.forward(middle);
    }
}
