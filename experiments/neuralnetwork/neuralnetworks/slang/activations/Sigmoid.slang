// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off
implementing NeuralNetworks;

__include IScalarActivation;
__include basetypes.DiffCoopVec;

namespace Activation
{

public struct Sigmoid<T : IReal, int Width> : IScalarActivation<T, Width>
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

public extension<T : IReal, int Width> Sigmoid<T, Width> : IModel<DiffCoopVec<T, Width>, DiffCoopVec<T, Width>>
{
    [BackwardDerivative(backward)]
    public DiffCoopVec<T, Width> forward(DiffCoopVec<T, Width> x)
    {
        return 1.0f / (1.0f + exp(-x));
    }

    public void backward(inout DifferentialPair<DiffCoopVec<T, Width>> x, DiffCoopVec<T, Width> grad)
    {
        let sigmoid = forward(x.p);
        let dSigmoid = sigmoid * (1.0f - sigmoid);
        x = diffPair(x.p, dSigmoid * grad);
    }
}

}
