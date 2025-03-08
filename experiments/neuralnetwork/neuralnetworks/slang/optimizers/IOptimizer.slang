// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
implementing NeuralNetworks;

__include basetypes.IReal;

public interface IOptimizerState<T : IReal>
{
    public __init(T param);
}

public interface IOptimizer<T : IReal>
{
    public associatedtype State : IOptimizerState<T>;

    public void step(inout State state, inout T param, inout T grad);
}
