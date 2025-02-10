// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
implementing slangpy;

public struct GridArg<let N : int> {

    public typealias VectorT = vector<int, N>;
    public typealias ArrayT = Array<int, N>;

    int[N] offset;
    int[N] stride;

    // Array load
    public void load(ContextND<N> context, out ArrayT value) {
        var t = context.call_id;
        for (int i = 0; i < N; i++) {
            value[i] = offset[i] + t[i] * stride[i];
        }
    }

    // Vector load (transpose of array load)
    public void load(ContextND<N> context, out VectorT value) {
        var t = context.call_id;
        int end = N - 1;
        for (int i = 0; i < N; i++) {
            value[end - i] = offset[i] + t[i] * stride[i];
        }
    }

    public void load(ContextND<1> context, out int value) {
        value = context.call_id[0];
    }
}
