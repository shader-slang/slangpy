// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
implementing slangpy;

public struct RandFloatArg<let WarmupIterations : int = 0> {
    public uint seed;
    public float min;
    public float max;

    // returns float in [0, 1)
    float hash_to_uniform_float(uint hash) {
        uint32_t k = (hash & 0x7FFFFF);       // 2^23 - 1
        return (float)k / (float)0x800000; // 2^23
    }
    vector<float,N> hash_to_uniform_vector<let N:int>(vector<uint,N> hash) {
        vector<uint,N> k = (hash & 0x7FFFFF);    // 2^23 - 1
        return k / (float)0x800000; // 2^23
    }

    public void load(Context0D context, out float value) {
        var hash = wang_hash_thread_scalar<WarmupIterations>(context.thread_id, seed);
        value = lerp(min, max, hash_to_uniform_float(hash));
    }
    public void store(Context0D context, in float value) {};

    public void load<let N : int>(Context0D context, out vector<float, N> value) {
        var hash = wang_hash_thread_vector<WarmupIterations, N>(context.thread_id, seed);
        value = lerp(min, max, hash_to_uniform_vector(hash));
    }
    public void store<let N : int>(Context0D context, in vector<float, N> value) {};
}
