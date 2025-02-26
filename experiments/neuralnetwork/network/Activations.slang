// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off

import NeuralModules;

namespace Activation
{

interface IActivation<T : IScalar, int Width>
{
    [BackwardDifferentiable, PreferRecompute] T activate(T x);
}
extension<T : IScalar, int Width, Act : IActivation<T, Width>> Act : IModule<T, Width, Width>
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

struct None<T : IScalar, int K> : IActivation<T, K>
{
    [BackwardDifferentiable, PreferRecompute]
    T activate(T x)
    {
        return x;
    }
};

struct ReLU<T : IScalar, int K> : IActivation<T, K>
{
    [BackwardDifferentiable, PreferRecompute]
    T activate(T x)
    {
        return max(x, T(0.0f));
    }
};

struct LeakyReLU<T : IScalar, int K> : IActivation<T, K>
{
    T negativeSlope;

    [BackwardDifferentiable, PreferRecompute]
    T activate(T x)
    {
        return max(x, T(0.0f)) + min(x, T(0.0f)) * negativeSlope;
    }
};

struct ELU<T : IScalar, int K> : IActivation<T, K>
{
    T a;

    [BackwardDifferentiable, PreferRecompute]
    T activate(T x)
    {
        return a * (exp(-min(x, T(0.0f))) - T(1.0f)) + max(x, T(0.0f));
    }
};

struct Swish<T : IScalar, int K> : IActivation<T, K>
{
    [BackwardDifferentiable, PreferRecompute]
    T activate(T x)
    {
        return x / (T(1.0f) + exp(-x));
    }
};

struct Tanh<T : IScalar, int K> : IActivation<T, K>
{
    [BackwardDifferentiable, PreferRecompute]
    T activate(T x)
    {
        return tanh(x);
    }
};

struct Sigmoid<T : IScalar, int K> : IActivation<T, K>
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

struct Exp<T : IScalar, int K> : IActivation<T, K>
{
    [BackwardDifferentiable, PreferRecompute]
    T activate(T x)
    {
        return exp(x); 
    }
};

}