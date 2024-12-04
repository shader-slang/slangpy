// SPDX-License-Identifier: Apache-2.0
import "slangpy";
import "wanghasharg";

struct RandFloatArg<let N : int>  {
    typealias T = vector<float, N>;

    uint seed;
    float min;
    float max;

    void load(Context0D context, out float value) {
        var tid = context.thread_id;
        var hash = wang_hash(wang_hash(tid.x) ^ seed);
        value = lerp(min, max, saturate((hash % 1000000) / 1000000.0f));
    }
    void store(Context0D context, in float value) {};

    void load(Context0D context, out T value) {
        var tid = context.thread_id;
        var hash = wang_hash(tid.x) ^ seed;
        for (int i = 0; i < N; i++) {
            hash = wang_hash(hash);
            value[i] = lerp(min,max, saturate((hash % 1000000) / 1000000.0f));
        }
    }
    void store(Context0D context, in T value) {};
}
