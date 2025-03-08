// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
implementing NeuralNetworks;

__include IOptimizer;

public struct FullPrecisionOptimizer<T : IReal, NestedOptimizer : IOptimizer<float>>
{
    public float gradientScale;
    public NestedOptimizer nestedOptim;
}

public extension<NestedOptimizer : IOptimizer<float>> FullPrecisionOptimizer<float, NestedOptimizer> : IOptimizer<float>
{
    public typealias State = NestedOptimizer::State;

    public void step(inout State state, inout float param, inout float grad)
    {
        nestedOptim.step(state, param, grad);
    }
}

// This struct definition should ideally move into the extension for T==half,
// (and avoid the NestedOptimizer generic param), but this triggers two separate
// reflection bugs (slang issue #6544, #6546)
public struct FullPrecisionParameterCopy<NestedOptimizer : IOptimizer<float>> : IOptimizerState<half>
{
    public float paramF;

    public NestedOptimizer::State nestedState;

    public __init(half paramH)
    {
        paramF = float(paramH);
        nestedState = NestedOptimizer::State(paramF);
    }
}
public extension<NestedOptimizer : IOptimizer<float>> FullPrecisionOptimizer<half, NestedOptimizer> : IOptimizer<half>
{
    public typealias State = FullPrecisionParameterCopy<NestedOptimizer>;

    public void step(inout State state, inout half paramH, inout half gradH)
    {
        float gradF = float(gradH) / gradientScale;
        nestedOptim.step(state.nestedState, state.paramF, gradF);
        gradH = half(gradF);
        paramH = half(state.paramF);
    }
}
