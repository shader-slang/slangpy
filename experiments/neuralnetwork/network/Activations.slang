// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off
import ArrayModule;

namespace Activation
{

interface IActivation<T : IReal, int Width>
{
    [BackwardDifferentiable, PreferRecompute] T activate(T x);
}
extension<T : IReal, int Width, Act : IActivation<T, Width>> Act : IArrayModule<T, Width, Width>
{
    [BackwardDifferentiable]
    T[Width] forward(T x[Width])
    {
        T y[Width];
        [ForceUnroll]
        for (int i = 0; i < Width; ++i)
            y[i] = activate(x[i]);
        return y;
    }
}

struct None<T : IReal, int K> : IActivation<T, K>
{
    [BackwardDifferentiable, PreferRecompute]
    T activate(T x)
    {
        return x;
    }
};

struct ReLU<T : IReal, int K> : IActivation<T, K>
{
    [BackwardDifferentiable, PreferRecompute]
    T activate(T x)
    {
        return max(x, T(0.0f));
    }
};

struct LeakyReLU<T : IReal, int K> : IActivation<T, K>
{
    T negativeSlope;

    [BackwardDifferentiable, PreferRecompute]
    T activate(T x)
    {
        return max(x, T(0.0f)) + min(x, T(0.0f)) * negativeSlope;
    }
};

struct ELU<T : IReal, int K> : IActivation<T, K>
{
    T a;

    [BackwardDifferentiable, PreferRecompute]
    T activate(T x)
    {
        return a * (exp(-min(x, T(0.0f))) - T(1.0f)) + max(x, T(0.0f));
    }
};

struct Swish<T : IReal, int K> : IActivation<T, K>
{
    [BackwardDifferentiable, PreferRecompute]
    T activate(T x)
    {
        return x / (T(1.0f) + exp(-x));
    }
};

struct Tanh<T : IReal, int K> : IActivation<T, K>
{
    [BackwardDifferentiable, PreferRecompute]
    T activate(T x)
    {
        return tanh(x);
    }
};

struct Sigmoid<T : IReal, int K> : IActivation<T, K>
{
    [BackwardDerivative(activate_bwd), PreferRecompute]
    T activate(T x)
    {
        return T(1.0f) / (T(1.0f) + exp(-x));
    }

    void activate_bwd(inout DifferentialPair<T> x, T.Differential grad)
    {
        let sigmoid = activate(x.p);
        let dSigmoid = sigmoid * (T(1.0f) - sigmoid);
        x = diffPair(x.p, T.dmul(dSigmoid, grad));
    }
};

struct Exp<T : IReal, int K> : IActivation<T, K>
{
    [BackwardDifferentiable, PreferRecompute]
    T activate(T x)
    {
        return exp(x); 
    }
};

}