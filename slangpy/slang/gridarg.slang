// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
implementing slangpy;

public struct GridArg<let N : int> {

    int[N] offset;
    int[N] stride;

    // Array load
    public void load<let N : int, T : __BuiltinIntegerType>(ContextND<N> context, out Array<T, N> value) {
        var t = context.call_id;
        int end = N - 1;
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

// Experimental way of expressing rules for what types can be
// passed as arguments to the load method of GridArg
struct GridArgConversion<SlangType, let Dim : int> {
}
extension<let N : int, T : __BuiltinIntegerType> GridArgConversion<Array<T, N>, N> {
    typealias ArgType = Array<T, N>;
}
extension<let N : int, T : __BuiltinIntegerType> GridArgConversion<Array<T, N>, -1> {
    typealias ArgType = Array<T, N>;
}
extension<let N : int, T : __BuiltinIntegerType> GridArgConversion<vector<T, N>, N> {
    typealias ArgType = vector<T, N>;
}
extension<let N : int, T : __BuiltinIntegerType> GridArgConversion<vector<T, N>, -1> {
    typealias ArgType = vector<T, N>;
}
extension<T : __BuiltinIntegerType> GridArgConversion<T, -1> {
    typealias ArgType = T;
}
extension<T : __BuiltinIntegerType> GridArgConversion<T, 1> {
    typealias ArgType = T;
}
