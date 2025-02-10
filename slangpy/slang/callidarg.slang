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
    public void load<let N : int>(ContextND<N> context, out vector<uint,N> value) {
        var t = context.call_id;
        int end = N - 1;
        for (int i = 0; i < N; i++) {
            value[end-i] = t[i];
        }
    }


}
