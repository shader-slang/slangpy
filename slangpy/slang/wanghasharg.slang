// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
implementing slangpy;

public uint wang_hash(uint seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

public vector<uint, N> wang_hash_thread_vector<let WarmupIterations : int, let N : int>(int3 thread_id, int seed) {
    var tid = thread_id;

    // Start with the thread id as the hash, scaled by N to avoid cross dimension
    // correlation, then do optional warmup
    var hash = tid.x * N;
    for (int i = 0; i < WarmupIterations; i++) {
        hash = wang_hash(hash);
    }

    // XOR with seed, then generate per component hashes
    vector<uint, N> value;
    hash = hash ^ seed;
    [ForceUnroll]
    for (int i = 0; i < N; i++) {
        hash = wang_hash(hash);
        value[i] = hash;
    }
    return value;
}

public uint wang_hash_thread_scalar<let WarmupIterations : int>(int3 thread_id, int seed) {
    var tid = thread_id;

    // Start with the thread id as the hash, then do optional warmup
    var hash = tid.x;
    for (int i = 0; i < WarmupIterations; i++) {
        hash = wang_hash(hash);
    }

    // XOR with seed, then generate result hash
    hash = hash ^ seed;
    return wang_hash(hash);
}


public struct WangHashArg<let WarmupIterations: int=0> {

    public uint seed;

    public void load<let N : int>(Context0D context, out vector<uint, N> value) {
        value = wang_hash_thread_vector<WarmupIterations, N>(context.thread_id, seed);
    }
    public void store<let N : int>(Context0D context, in vector<uint, N> value) {};

    public void load(Context0D context, out uint value) {
        value = wang_hash_thread_scalar<WarmupIterations>(context.thread_id, seed);
    }
    public void store(Context0D context, in uint value) {};
}
