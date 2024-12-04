// SPDX-License-Identifier: Apache-2.0
import "slangpy";

uint wang_hash(uint seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

struct WangHashArg<let N : int> {
    typealias T = vector<uint, N>;

    uint seed;

    void load(Context0D context, out T value) {
        var tid = context.thread_id;
        var hash = tid.x ^ seed; 
        for (int i = 0; i < N; i++) {
            hash = wang_hash(hash);
            value[i] = hash;
        }
    }
    void store(Context0D context, in T value) {};
}
