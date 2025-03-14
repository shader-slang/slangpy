// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off
implementing NeuralNetworks;

__include IScalarActivation;
__include basetypes.DiffCoopVec;

namespace Activation
{

public struct SmeLU<T : IReal, int Width> : IScalarActivation<T, Width>
{
    /* Linear ramp (slope Beta) for x < -Alpha.
       Linear ramp (slope 1) for x > Alpha.
       Smooth quadratic interpolant for -Alpha < x < Alpha.
       A, B, C are the coefficients of the quadratic A*x^2 + B*x + C. */
    static const T Alpha = T(0.5f);
    static const T Beta = T(0.01f);
    static const T A = (T(1.f) - Beta) / (T(4.f) * Alpha);
    static const T B = (T(1.f) + Beta) / T(2.f);
    static const T C = Alpha * (T(1.f) - Beta) / T(4.f);

    [BackwardDifferentiable, PreferRecompute]
    public T activate(T x)
    {
        let left = max(Beta * x, x + Alpha * (T(1.f) - Beta));
        let middle = A * x * x + B * x + C;
        let right = max(x, Alpha);

        return min(left, min(middle, right));
    }
}

public extension<T : IReal, int Width> SmeLU<T, Width> : IModel<DiffCoopVec<T, Width>, DiffCoopVec<T, Width>>
{
    typealias CV = DiffCoopVec<T, Width>;
    
    [BackwardDerivative(backward)]
    public DiffCoopVec<T, Width> forward(DiffCoopVec<T, Width> x)
    {
        let left = max(Beta * x, x + Alpha * (T(1.f) - Beta));
        let middle = A * x * x + C * x + C;
        let right = max(x, CV(Alpha));
        
        return min(left, min(middle, right));
    }
    
    public void backward(inout DifferentialPair<DiffCoopVec<T, Width>> x, DiffCoopVec<T, Width> grad)
    {
        let left = max(x.p + Alpha + Beta, CV(Beta));
        let middle = max(Beta - Alpha - x.p, T(2.f) * A * x.p + B);
        let right = CV(1.f);
        
        let dSmeLU = min(left, min(middle, right));
        x = diffPair(x.p, dSmeLU * grad);
    }
}

}
