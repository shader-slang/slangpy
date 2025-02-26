// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import NeuralModules;

struct LinearLayer<T : IScalar, int NumInputs, int NumOutputs> : IModule<T, NumInputs, NumOutputs> where T.Differential : IArithmeticAtomicable
{
    StructuredBuffer<T> weights, biases;
    RWStructuredBuffer<Atomic<T.Differential>> weightGrads, biasGrads;

    [BackwardDerivative(backward)]
    T[NumOutputs] forward(T x[NumInputs])
    {
        T y[NumOutputs];
        for (int row = 0; row < NumOutputs; ++row)
        {
            var sum = biases[row];
            [ForceUnroll]
            for (int col = 0; col < NumInputs; ++col)
                sum += weights[row * NumInputs + col] * x[col];

            y[row] = sum;
        }

        return y;
    }

    void backward(inout DifferentialPair<T[NumInputs]> x, T.Differential grad[NumOutputs])
    {
        var dx = T[NumInputs].dzero();
        for (int row = 0; row < NumOutputs; ++row)
        {
            biasGrads[row].add(grad[row]);

            [ForceUnroll]
            for (int col = 0; col < NumInputs; ++col)
            {
                weightGrads[row * NumInputs + col].add(T.dmul(x.p[col], grad[row]));
                dx[col] = T.dadd(dx[col], T.dmul(weights[row * NumInputs + col], grad[row]));
            }
        }
        x = diffPair(x.p, dx);
    }
}
