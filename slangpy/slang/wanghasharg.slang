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

public struct WangHashArg {

    public uint seed;
    
    public void load<let N : int>(Context0D context, out vector<uint,N> value) {
        var tid = context.thread_id;
        var hash = tid.x ^ seed; 
        for (int i = 0; i < N; i++) {
            hash = wang_hash(hash);
            value[i] = hash;
        }
    }
    public void store<let N : int>(Context0D context, in vector<uint, N> value) {};
}
