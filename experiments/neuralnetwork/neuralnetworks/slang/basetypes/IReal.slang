// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off

implementing NeuralNetworks;

// Helper typedef for an interface that is constrained to a differentiable, scalar element type
// Currently implemented by half, float and double, though subject to change
// This is used in many model implementations and slang's internal interface for this is a bit unwieldy,
// so we define a nicer name centrally here
public typealias IReal = __BuiltinFloatingPointType;

public interface IRealArray<T : IReal, let D : int> : IDifferentiable
{
    public __subscript(int index)->T { get; set; }
}
    
// Array and vector conform to array interface
public extension<T : IReal, int D> Array<T, D> : IRealArray<T, D> { public __subscript(int index)->T { get; set; } }
public extension<T : IReal, int D> vector<T, D> : IRealArray<T, D> { public __subscript(int index)->T { get; set; } }
