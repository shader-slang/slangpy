// SPDX-License-Identifier: Apache-2.0
implementing slangpy;

public struct RandFloatArg<let N : int> {
    public typealias T = vector<float, N>;

    public uint seed;
    public float min;
    public float max;

    public void load(Context0D context, out float value) {
        var tid = context.thread_id;
        var hash = wang_hash(wang_hash(tid.x) ^ seed);
        value = lerp(min, max, saturate((hash % 1000000) / 1000000.0f));
    }
    public void store(Context0D context, in float value) {};

    public void load(Context0D context, out T value) {
        var tid = context.thread_id;
        var hash = wang_hash(tid.x) ^ seed;
        for (int i = 0; i < N; i++) {
            hash = wang_hash(hash);
            value[i] = lerp(min,max, saturate((hash % 1000000) / 1000000.0f));
        }
    }
    public void store(Context0D context, in T value) {};
}
