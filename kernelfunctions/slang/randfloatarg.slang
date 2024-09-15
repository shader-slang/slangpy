import "slangpy";
import "wanghasharg";

struct RandFloatArg<let N : int> : IValueCallData<vector<float, N>> {
    typealias primal_type = vector<float, N>;

    uint seed;
    float min;
    float max;

    void load_primal(IContext context, out primal_type value) {
        var tid = context.get_thread_id();
        var hash = tid.x ^ seed;
        for (int i = 0; i < N; i++) {
            hash = wang_hash(hash);
            value[i] = lerp(min,max, saturate((hash % 1000000) / 1000000.0f));
        }
    }
    void store_primal(IContext context, in primal_type value) {};
}
