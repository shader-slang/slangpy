// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off

__exported import DiffCoopVec;
__exported import CoopVecModule;

namespace CoopVecAct
{

interface IActivation<T : IReal, int Width> : ICoopVecModule<T, Width, Width>
{
    [BackwardDifferentiable]
    DiffCoopVec<T, Width> forward(DiffCoopVec<T, Width> x);
}
extension<T : IReal, int Width, Act : IActivation<T, Width>> Act
{
    typealias IOType = InputType;
}

struct None<T : IReal, int Width> : IActivation<T, Width>
{
    [BackwardDifferentiable]
    IOType forward(IOType x)
    {
        return x;
    }
};

struct ReLU<T : IReal, int Width> : IActivation<T, Width>
{
    [BackwardDifferentiable, PreferRecompute]
    IOType forward(IOType x)
    {
        return max(x, IOType(0.0f)); 
    }
};

struct LeakyReLU<T : IReal, int Width> : IActivation<T, Width>
{
    T negativeSlope;

    [BackwardDifferentiable, PreferRecompute]
    IOType forward(IOType x)
    {
        return max(x, IOType(0.0f)) - max(-x, IOType(0.0f)) * negativeSlope;
    }
};

struct ELU<T : IReal, int Width> : IActivation<T, Width>
{
    T a;

    [BackwardDifferentiable]
    IOType forward(IOType x)
    {
        return a * (exp(max(-x, IOType(0.0f))) - 1.0f) + max(x, IOType(0.0f));
    }
};

struct Swish<T : IReal, int Width> : IActivation<T, Width>
{
    [BackwardDifferentiable]
    IOType forward(IOType x)
    {
        return x / (1.0f + exp(-x));
    }
};

struct Tanh<T : IReal, int Width> : IActivation<T, Width>
{
    [BackwardDifferentiable]
    IOType forward(IOType x)
    {
        return tanh(x);
    }
};

struct Sigmoid<T : IReal, int Width> : IActivation<T, Width>
{
    [BackwardDerivative(backward)]
    IOType forward(IOType x)
    {
        return 1.0f / (1.0f + exp(-x));
    }

    void backward(inout DifferentialPair<IOType> x, IOType grad)
    {
        let sigmoid = forward(x.p);
        let dSigmoid = sigmoid * (1.0f - sigmoid);
        x = diffPair(x.p, dSigmoid * grad);
    }
};

struct Exp<T : IReal, int Width> : IActivation<T, Width>
{
    [BackwardDifferentiable, PreferRecompute]
    IOType forward(IOType x)
    {
        return exp(x); 
    }
};

}