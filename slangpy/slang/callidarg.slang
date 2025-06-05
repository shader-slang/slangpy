// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
implementing slangpy;

public struct CallIdArg {

    // Array load
    public void load<let N : int, T: __BuiltinIntegerType>(ContextND<N> context, out Array<T, N> value) {
        var t = context.call_id;
        for (int i = 0; i < N; i++) {
            value[i] = (T)t[i];
        }
    }

    // Vector load (transpose of array load)
    public void load<let N : int, T : __BuiltinIntegerType>(ContextND<N> context, out vector<T,N> value) {
        var t = context.call_id;
        int end = N - 1;
        for (int i = 0; i < N; i++) {
            value[end-i] = (T)t[i];
        }
    }

    //Vector load (transpose of array load)
    //TODO: Add array version of this, and see if we can support a version for any other size, not just N+1
    public void load<let N : int, T : __BuiltinIntegerType>(ContextND<N> context, out vector<T,N+1> value) {
        var t = context.call_id;
        int end = N - 1;
        for (int i = 0; i < N; i++) {
            value[end-i] = (T)t[i];
        }
        value[N] = (T)0; // Fill the last element with zero
    }

    // Scalar load (works for N = 1)
    public void load<T : __BuiltinIntegerType>(ContextND<1> context, out T value) {
        value = (T)context.call_id[0];
    }
}

// Rules for how to vectorize a Python call id argument to a given Slang type.
// Dim is user specified Python side, -1 means use the same dimensionality as the Slang type.
struct VectorizeCallidArgTo<SlangParameterType, let Dim: int> {
}
extension<let N : int, T : __BuiltinIntegerType> VectorizeCallidArgTo<Array<T,N>, N> {
    typealias VectorType = Array<T,N>;
}
extension<let N : int, T : __BuiltinIntegerType> VectorizeCallidArgTo<Array<T, N>, -1>  {
    typealias VectorType = Array<T, N>;
}
extension<let N : int, T : __BuiltinIntegerType> VectorizeCallidArgTo<vector<T,N>, N> {
    typealias VectorType = vector<T, N>;
}
extension<let N : int, T : __BuiltinIntegerType> VectorizeCallidArgTo<vector<T, N>, -1> {
    typealias VectorType = vector<T, N>;
}
extension<T : __BuiltinIntegerType> VectorizeCallidArgTo<T, -1> {
    typealias VectorType = T;
}
extension<T : __BuiltinIntegerType> VectorizeCallidArgTo<T, 1> {
    typealias VectorType = T;
}

// Functions provided for use in user's shaders
// Note: In the returned arrays, the first component represents the right-most dimension,
// the second component the next dimension to the left, and so on. This aligns with
// behavior described at https://slangpy.shader-slang.org/en/latest/src/generators/generator_ids.html
// but could be confusing to a user using these and expecting [x, y, z, ...] formatting.
export public int[N] get_call_id<let N: int>();
export public uint[N] get_call_group_id<let N: int>();
export public uint[N] get_call_group_thread_id<let N: int>();
