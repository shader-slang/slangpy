import "slangpy";

struct ThreadIdArg<let N : int> {
    typealias T = vector<uint, N>;
    void load(Context0D context, out T value) {
        var t = context.thread_id;
        for (int i = 0; i < N; i++) {
            value[i] = t[i];
        }
    }
    void store(Context0D context, in T value) {};
}
