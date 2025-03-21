// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
implementing NeuralNetworks;

__include basetypes.IModel;
__include basetypes.DiffCoopVec;

/*
    Represents a linear neural network layer, i.e. a matrix-vector multiply/add A * x + b

    Takes an input with NumInputs elements, and returns NumOutputs elements.

    LinearLayer implements the multiply in software for plain arrays,
    and CoopVecLinearLayer uses hardware acceleration using cooperative vectors.
*/
public struct LinearLayer<
    T : IReal,
    int NumInputs,
    int NumOutputs
> : IModel<T[NumInputs], T[NumOutputs]> where T.Differential : IArithmeticAtomicable
{
    public StructuredBuffer<T> weights, biases;
    public RWStructuredBuffer<Atomic<T.Differential>> weightGrads, biasGrads;

    [BackwardDerivative(backward)]
    public OutputType forward(InputType x)
    {
        OutputType y;
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

    public void backward(inout DifferentialPair<InputType> x, OutputType.Differential grad)
    {
        var dx = InputType.dzero();
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

public struct CoopVecLinearLayer<
    T : IReal,
    int NumInputs,
    int NumOutputs
> : IModel<DiffCoopVec<T, NumInputs>, DiffCoopVec<T, NumOutputs>>
{
    static const CoopVecComponentType ComponentType = InputType::ComponentType;

    public StructuredBuffer<T> weights, biases;
    public RWStructuredBuffer<T> weightGrads, biasGrads;

    [BackwardDerivative(backward)]
    public OutputType forward(InputType x)
    {
        return OutputType(coopVecMatMulAdd<T, NumOutputs>(
            x.cv, ComponentType,
            weights, 0, ComponentType,
            biases, 0, ComponentType,
            CoopVecMatrixLayout.TrainingOptimal, false, 0
        ));
    }

    public void backward(inout DifferentialPair<InputType> x, OutputType grad)
    {
        coopVecOuterProductAccumulate(grad.cv, x.p.cv, weightGrads, 0, 0, CoopVecMatrixLayout.TrainingOptimal, ComponentType);
        coopVecReduceSumAccumulate(grad.cv, biasGrads, 0);

        let dX = coopVecMatMul<T, NumInputs>(grad.cv, ComponentType, weights, 0, ComponentType, CoopVecMatrixLayout.TrainingOptimal, true, 0);

        x = diffPair(x.p, InputType(dX));
    }
}
