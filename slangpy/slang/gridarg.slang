// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
implementing slangpy;

public struct GridArg<let N : int> {

    int[N] offset;
    int[N] stride;

    // Array load
    public void load<let N : int, T : __BuiltinIntegerType>(ContextND<N> context, out Array<T, N> value) {
        var t = context.call_id;
        for (int i = 0; i < N; i++) {
            value[i] = (T)(offset[i] + t[i] * stride[i]);
        }
    }

    // Vector load (transpose of array load)
    public void load<let N : int, T : __BuiltinIntegerType>(ContextND<N> context, out vector<T, N> value) {
        var t = context.call_id;
        int end = N - 1;
        for (int i = 0; i < N; i++) {
            value[end - i] = (T)(offset[i] + t[i] * stride[i]);
        }
    }

    // Scalar load (works for N = 1)
    public void load<T : __BuiltinIntegerType>(ContextND<1> context, out T value) {
        value = (T)context.call_id[0];
    }
}

// Rules for how to vectorize a Python grid argument to a given Slang type.
// Dim is user specified Python side, -1 means use the same dimensionality as the Slang type.
struct VectorizeGridArgTo<SlangParameterType, let Dim : int> {
}
extension<let N : int, T : __BuiltinIntegerType> VectorizeGridArgTo<Array<T, N>, N> {
    typealias VectorType = Array<T, N>;
}
extension<let N : int, T : __BuiltinIntegerType> VectorizeGridArgTo<Array<T, N>, -1> {
    typealias VectorType = Array<T, N>;
}
extension<let N : int, T : __BuiltinIntegerType> VectorizeGridArgTo<vector<T, N>, N> {
    typealias VectorType = vector<T, N>;
}
extension<let N : int, T : __BuiltinIntegerType> VectorizeGridArgTo<vector<T, N>, -1> {
    typealias VectorType = vector<T, N>;
}
extension<T : __BuiltinIntegerType> VectorizeGridArgTo<T, -1> {
    typealias VectorType = T;
}
extension<T : __BuiltinIntegerType> VectorizeGridArgTo<T, 1> {
    typealias VectorType = T;
}
