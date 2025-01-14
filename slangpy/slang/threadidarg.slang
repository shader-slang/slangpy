// SPDX-License-Identifier: Apache-2.0
implementing slangpy;

public struct ThreadIdArg<let N : int> {
    public typealias T = vector<uint, N>;
    public void load(Context0D context, out T value) {
        var t = context.thread_id;
        for (int i = 0; i < N; i++) {
            value[i] = t[i];
        }
    }
    public void store(Context0D context, in T value) {};
}
