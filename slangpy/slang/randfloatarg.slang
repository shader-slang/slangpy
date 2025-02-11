// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
implementing slangpy;

public struct RandFloatArg<let WarmupIterations : int = 0> {
    public uint seed;
    public float min;
    public float max;

    public void load(Context0D context, out float value) {
        var hash = wang_hash_thread_scalar<WarmupIterations>(context.thread_id, seed);
        value = lerp(min, max, saturate((hash % 1000000) / (1000000.0f-1.0f)));
    }
    public void store(Context0D context, in float value) {};

    public void load<let N : int>(Context0D context, out vector<float, N> value) {
        var hash = wang_hash_thread_vector<WarmupIterations, N>(context.thread_id, seed);
        value = lerp(min, max, saturate((hash % 1000000) / (1000000.0f - 1.0f)));
    }
    public void store<let N : int>(Context0D context, in vector<float, N> value) {};
}
