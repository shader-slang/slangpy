import "slangpy";
import "wanghasharg";

struct RandFloatArg<let N : int>  {
    typealias T = vector<float, N>;

    uint seed;
    float min;
    float max;

    void load_primal(IContext context, out float value) {
        var tid = context.get_thread_id();
        var hash = wang_hash(wang_hash(tid.x) ^ seed);
        value = lerp(min, max, saturate((hash % 1000000) / 1000000.0f));
    }
    void store_primal(IContext context, in float value) {};

    void load_primal(IContext context, out T value) {
        var tid = context.get_thread_id();
        var hash = wang_hash(tid.x) ^ seed;
        for (int i = 0; i < N; i++) {
            hash = wang_hash(hash);
            value[i] = lerp(min,max, saturate((hash % 1000000) / 1000000.0f));
        }
    }
    void store_primal(IContext context, in T value) {};
}
