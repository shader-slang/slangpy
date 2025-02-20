// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
__exported import CoopVecModule;

struct CoopVecLinearLayer<T : IReal, int FanIn, int FanOut> : ICoopVecModule<T, FanIn, FanOut>
{
    StructuredBuffer<T> parameters;
    RWStructuredBuffer<T> gradients;
    int2 offsets;

    [BackwardDerivative(backward)]
    OutputType forward(InputType x)
    {
        return OutputType(coopVecMatMulAdd<T, FanOut>(
            x.cv, ComponentType,
            parameters, offsets[0], ComponentType,
            parameters, offsets[1], ComponentType,
            CoopVecMatrixLayout.TrainingOptimal, false, 0
        ));
    }

    void backward(inout DifferentialPair<InputType> x, OutputType grad)
    {
        coopVecOuterProductAccumulate(grad.cv, x.p.cv, gradients, offsets[0], 0, CoopVecMatrixLayout.TrainingOptimal, ComponentType);
        coopVecReduceSumAccumulate(grad.cv, gradients, offsets[1]);

        let dX = coopVecMatMul<T, FanIn>(grad.cv, ComponentType, parameters, offsets[0], ComponentType, CoopVecMatrixLayout.TrainingOptimal, true, 0);

        x = diffPair(x.p, InputType(dX));
    }
}
