import "slangpy";

struct ThreadIdArg<let N : int> {
    typealias T = vector<uint, N>;
    void load_primal(Context0D context, out T value) {
        var t = context.thread_id;
        for (int i = 0; i < N; i++) {
            value[i] = t[i];
        }
    }
    void store_primal(Context0D context, in T value) {};
}
