import "slangpy";

struct ThreadIdArg<let N : int> : IValueCallData<vector<uint, N>> {
    typealias primal_type = vector<uint, N>;
    void load_primal(IContext context, out primal_type value) {
        var t = context.get_thread_id();
        for (int i = 0; i < N; i++) {
            value[i] = t[i];
        }
    }
    void store_primal(IContext context, in primal_type value) {};
}
