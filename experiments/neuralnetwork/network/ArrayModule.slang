// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import NeuralModules;

typealias IReal = __BuiltinFloatingPointType;

interface IArrayModule<T : IReal, int NumInputs, int NumOutputs> : IModule<T[NumInputs], T[NumOutputs]>
{
    [BackwardDifferentiable]
    T[NumOutputs] forward(T x[NumInputs]);
}

// Extensions:
// Anything that takes two arrays is also an array module (e.g. chaining two array modules)
/*extension<T : IReal, int NumInputs, int NumOutputs, Module : IModule<T[NumInputs], T[NumOutputs]>> Module : IArrayModule<T, NumInputs, NumOutputs>
{
    [BackwardDifferentiable]
    T[NumOutputs] forward(T x[NumInputs]);
}*/
// An array module can also be used as a vector module
extension<T : IReal, int NumInputs, int NumOutputs, Module : IModule<T[NumInputs], T[NumOutputs]>> Module : IModule<vector<T, NumInputs>, vector<T, NumOutputs>>
{
    [BackwardDifferentiable]
    vector<T, NumOutputs> forward(vector<T, NumInputs> x)
    {
        T xArray[NumInputs];
        [ForceUnroll]
        for (int i = 0; i < NumInputs; ++i)
            xArray[i] = x[i];

        let yArray = forward(xArray);

        vector<T, NumOutputs> y;
        [ForceUnroll]
        for (int i = 0; i < NumOutputs; ++i)
            y[i] = yArray[i];

        return y;
    }
}
