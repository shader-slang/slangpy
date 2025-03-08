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
    typealias InputType = InputT;
    typealias OutputType = OutputT;
}
// If input and output type happen to be identical, we also typedef an IOType to avoid confusion
public extension<T : IDifferentiable, Model : IModel<T, T>> Model
{
    typealias IOType = T;
}
