import "slangpy";

struct ThreadIdArg<let N : int> {
    typealias T = vector<uint, N>;
    void load_primal(IContext context, out T value) {
        var t = context.get_thread_id();
        for (int i = 0; i < N; i++) {
            value[i] = t[i];
        }
    }
    void store_primal(IContext context, in T value) {};
}
