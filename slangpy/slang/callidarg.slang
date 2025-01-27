// SPDX-License-Identifier: Apache-2.0
implementing slangpy;

public struct CallIdArg<let N : int> {
    public typealias T = vector<uint, N>;
    public void load(ContextND<N> context, out T value) {
        var t = context.call_id;
        for (int i = 0; i < N; i++) {
            value[i] = t[i];
        }
    }
    public void load(ContextND<1> context, out uint value) {
        value = context.call_id[0];
    }
}
