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

    void load_primal(IContext context, out T value) {
        var tid = context.get_thread_id();
        var hash = tid.x ^ seed; 
        for (int i = 0; i < N; i++) {
            hash = wang_hash(hash);
            value[i] = hash;
        }
    }
    void store_primal(IContext context, in T value) {};
}
