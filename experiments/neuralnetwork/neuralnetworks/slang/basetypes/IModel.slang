// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off

implementing NeuralNetworks;

// Base class for all trainable models
// Requirements are a differential input type and output type,
// and a differentiable forward method that transforms InputT to OutputT
public interface IModel<InputT : IDifferentiable, OutputT : IDifferentiable>
{
    [BackwardDifferentiable]
    public OutputT forward(InputT x);
}

// Extension that adds helper typedefs InputType and OutputType to all model implementations
public extension<InputT : IDifferentiable, OutputT : IDifferentiable, Model : IModel<InputT, OutputT>> Model
{
    public typealias InputType = InputT;
    public typealias OutputType = OutputT;
}
// If input and output type happen to be identical, we also typedef an IOType to avoid confusion
public extension<T : IDifferentiable, Model : IModel<T, T>> Model
{
    public typealias IOType = T;
}

// This is used by IModel.py to do type checking of slang code. Not intended to be used by user code.
namespace impl
{
    public void returnTypeWitness<InputT : IDifferentiable, OutputT : IDifferentiable, Model : IModel<InputT, OutputT>>() {}
}
