// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
implementing slangpy;

public struct CallIdArg {

    // Array load
    public void load<let N : int, T: __BuiltinIntegerType>(ContextND<N> context, out Array<T, N> value) {
        var t = context.call_id;
        int end = N - 1;
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

    // Scalar load (works for N = 1)
    public void load<T : __BuiltinIntegerType>(ContextND<1> context, out T value) {
        value = (T)context.call_id[0];
    }    
}

// Set of generic rules to describe type conversion from call id arg to a slang type.
struct CallIdArgConversion<SlangType, let Dim: int> {
}
extension<let N : int, T : __BuiltinIntegerType> CallIdArgConversion<Array<T,N>, N> {
    typealias ArgType = Array<T,N>;
}
extension<let N : int, T : __BuiltinIntegerType> CallIdArgConversion<Array<T, N>, -1>  {
    typealias ArgType = Array<T, N>;
}
extension<let N : int, T : __BuiltinIntegerType> CallIdArgConversion<vector<T,N>, N> {
    typealias ArgType = vector<T, N>;
}
extension<let N : int, T : __BuiltinIntegerType> CallIdArgConversion<vector<T, N>, -1> {
    typealias ArgType = vector<T, N>;
}
extension<T : __BuiltinIntegerType> CallIdArgConversion<T, -1> {
    typealias ArgType = T;
}
extension<T : __BuiltinIntegerType> CallIdArgConversion<T, 1> {
    typealias ArgType = T;
}
