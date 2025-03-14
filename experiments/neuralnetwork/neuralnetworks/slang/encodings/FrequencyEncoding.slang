// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off
implementing NeuralNetworks;

__include basetypes.IModel;

/*
    Performs a frequency encoding of the input.
    
    Each element x of the input is mapped to a series of sines and cosines of increasing
    frequency, i.e. sin(x), cos(x), sin(2*x), cos(2*x), sin(4*x), cos(4*x), ...

    Each input element is turned into 2 * num_octaves outputs.

    Expects a plain array as input, and returns a plain array.
*/
public struct FrequencyEncoding<T : IReal, int NumInputs, int NumOctaves> : IModel<T[NumInputs], T[NumOctaves * NumInputs * 2]>
{
    [BackwardDifferentiable]
    public OutputType forward(InputType x)
    {
        OutputType y;
        
        [ForceUnroll]
        for (int i = 0; i < NumInputs; ++i)
        {
            const int base = i * NumOctaves * 2;

            T sn, cn;
            sincos(x[i] * T.getPi(), sn, cn);
            y[base + 0] = sn;
            y[base + 1] = cn;
            [ForceUnroll]
            for (int j = 1; j < NumOctaves; ++j) {
                sn = T(2.0f) * sn * cn;
                cn = T(2.0f) * cn * cn - T(1.0f);
                y[base + j * 2 + 0] = sn;
                y[base + j * 2 + 1] = cn;
            }
        }

        return y;
    }
}
