// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
implementing NeuralNetworks;

__include IOptimizer;

public struct AdamOptimizer<T : IReal> : IOptimizer<T>
{
    public struct AdamState<S : IReal> : IOptimizerState<S>
    {
        S mean;
        S variance;

        public __init(S param)
        {
            mean = S(0.0f);
            variance = S(-1.0f);
        }
    }
    public typealias State = AdamState<T>;

    public T beta1;
    public T beta2;
    public T epsilon;
    public T learningRate;

    public void step(inout State state, inout T param, inout T grad)
    {
        bool isFirstIteration = state.variance < T(0.0f);
        T blendMean = isFirstIteration ? T(0.0f) : beta1;
        T blendVariance = isFirstIteration ? T(0.0f) : beta2;

        state.mean = lerp(grad, state.mean, blendMean);
        state.variance = lerp(grad * grad, state.variance, blendVariance);

        param -= learningRate * state.mean / (sqrt(state.variance) + epsilon);
        grad = T(0.0f);
    }
}
