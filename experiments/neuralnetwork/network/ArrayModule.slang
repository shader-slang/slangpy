// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
__exported import NeuralModules;

interface IArrayModule<T : IReal, int NumInputs, int NumOutputs> : IModule<T[NumInputs], T[NumOutputs]>
{
    [BackwardDifferentiable]
    T[NumOutputs] forward(T x[NumInputs]);
}

struct ConvertPrecision<InT : IReal, OutT : IReal, int Width> {}

extension<InT : IReal, OutT : IReal, int Width> ConvertPrecision<InT, OutT, Width> : IModule<InT[Width], OutT[Width]>
{
    [BackwardDifferentiable]
    OutputType forward(InputType x)
    {
        OutputType y;
        [ForceUnroll]
        for (int i = 0; i < Width; ++i)
            y[i] = OutT(x[i].toFloat());
        return y;
    }
}
extension<T : IReal, int Width> ConvertPrecision<T, T, Width> : IModule<T[Width], T[Width]>
{
    [BackwardDifferentiable]
    OutputType forward(InputType x)
    {
        return x;
    }
}

// Extensions:
// Anything that takes arrays can also take vectors
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
