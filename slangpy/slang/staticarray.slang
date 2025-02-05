// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Shared implementation for anything that implements IArray and has a statically known size
implementing slangpy;

public interface ISizedArray<T, let D : int> { __subscript(int index)->T { get; } }

// Array, vector and matrix conform to ISizedArray
public extension<T, let D : int> Array<T, D> : ISizedArray<T, D> {}
public extension<T, let D : int> vector<T, D> : ISizedArray<T, D> {}
public extension<T, let R : int, let C : int, let L : int> matrix<T, R, C, L> : ISizedArray<vector<T, C>, R> {}
