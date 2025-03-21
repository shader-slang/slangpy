// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
implementing NeuralNetworks;

__include IOptimizer;

public struct AdamOptimizer<T : IReal> : IOptimizer<T>
{
    // Internal state storing per-parameter moments
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

    // Adam parameters
    public T beta1;
    public T beta2;
    public T epsilon;
    public T learningRate;

    public void step(inout State state, inout T param, inout T grad)
    {
        // Note: The standard Adam implementation contains bias in the mean and variance
        // This is because they are a moving average and are initialized to zero,
        // leading to a bias towards zero that diminishes over time.
        // The original Adam paper suggests to cancel out the bias by dividing out
        // the weight of the zero term, computed with a power of the betas (!) and the
        // iteration count.
        // This implements a much simpler solution to cancelling the bias by initializing
        // the moving average from the gradients in the first iteration, instead of zero.
        // This avoids the bias entirely.
        // We avoid an iteration count or separate boolean flag by initializing the variance
        // part of the moments to a negative number; in normal operation, the variance can
        // only be positive, and we know we're starting from fresh initialization if the
        // variance is negative.
        bool isFirstIteration = state.variance < T(0.0f);
        T blendMean = isFirstIteration ? T(0.0f) : beta1;
        T blendVariance = isFirstIteration ? T(0.0f) : beta2;

        state.mean = lerp(grad, state.mean, blendMean);
        state.variance = lerp(grad * grad, state.variance, blendVariance);

        param -= learningRate * state.mean / (sqrt(state.variance) + epsilon);
        grad = T(0.0f);
    }
}
