// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off

import NeuralModules;

struct FrequencyEncoding<T : IScalar, int NumInputs, int NumScales> : IModule<T, NumInputs, NumScales * NumInputs * 2>
{
    [BackwardDifferentiable]
    T[NumScales * NumInputs * 2] forward(T x[NumInputs])
    {
        T y[NumScales * NumInputs * 2];
        
        [ForceUnroll]
        for (int i = 0; i < NumInputs; ++i)
        {
            const int base = i * NumScales * 2;

            T sn, cn;
            sincos(x[i] * T.getPi(), sn, cn);
            y[base + 0] = sn;
            y[base + 1] = cn;
            [ForceUnroll]
            for (int j = 1; j < NumScales; ++j) {
                sn = T(2.0f) * sn * cn;
                cn = T(2.0f) * cn * cn - T(1.0f);
                y[base + j * 2 + 0] = sn;
                y[base + j * 2 + 1] = cn;
            }
        }

        return y;
    }
}
