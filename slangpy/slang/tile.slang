// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
implementing slangpy;

public interface ITile<T, let N : int> {
    public bool load(vector<int, N> call_id_offset, out T value);
}

public interface IRWTile<T, let N : int> {
    public bool load(vector<int, N> call_id_offset, out T value);
    public bool store(vector<int, N> call_id_offset, in T value);
}

public struct Tile<ContainerT : IIndexableType> : ITile<ContainerT.IndexType, ContainerT.Dims>, IRWTile<ContainerT.IndexType, ContainerT.Dims> {
    public ContainerT data;
    public ContextND<ContainerT.Dims> context;
    public int[ContainerT.Dims] shape;

    // Creates a context with a given grid offset, checking for out-of-bounds at the same time.
    bool create_sub_context(int[ContainerT.Dims] call_id_offset, out ContextND<ContainerT.Dims> sub_context) {
        sub_context.thread_id = context.thread_id;
        bool oob = false;
        for (int i = 0; i < ContainerT.Dims; i++) {
            sub_context.call_id[i] = context.call_id[i] + call_id_offset[i];
            oob = oob || sub_context.call_id[i] < 0 || sub_context.call_id[i] >= shape[i];
        }
        return !oob;
    }

    // Same as array create context, but transposes the vector before adding it to the context.
    bool create_sub_context(vector<int, ContainerT.Dims> call_id_offset, out ContextND<ContainerT.Dims> sub_context) {
        sub_context.thread_id = context.thread_id;
        bool oob = false;
        int end = ContainerT.Dims - 1;
        for (int i = 0; i < ContainerT.Dims; i++) {
            sub_context.call_id[i] = context.call_id[i] + call_id_offset[end - i];
            oob = oob || sub_context.call_id[i] < 0 || sub_context.call_id[i] >= shape[i];
        }
        return !oob;
    }

    public bool load(vector<int, ContainerT.Dims> call_id_offset, out ContainerT.IndexType value) {
        ContextND<ContainerT.Dims> sub_context;
        if (create_sub_context(call_id_offset, sub_context)) {
            data.load(sub_context, value);
            return true;
        } else {
            return false;
        }
    }

    public bool store(vector<int, ContainerT.Dims> call_id_offset, in ContainerT.IndexType value) {
        ContextND<ContainerT.Dims> sub_context;
        if (create_sub_context(call_id_offset, sub_context)) {
            data.store(sub_context, value);
            return true;
        } else {
            return false;
        }

    }

    public typealias SPType = TileType<ContainerT>;
    public typealias SPDims = Size<ContainerT.Dims>;
}

public struct TileType<ContainerT : IIndexableType>  {
    public ContainerT data;
    public int[ContainerT.Dims] stride;
    public int[ContainerT.Dims] shape;

    public void load(ContextND<ContainerT.Dims> context, out Tile<ContainerT> value) {
        value.data = data;
        value.context.thread_id = context.thread_id;
        for (int i = 0; i < ContainerT.Dims; i++) {
            value.context.call_id[i] = context.call_id[i] * stride[i];
            value.shape[i] = shape[i];
        }
    }
    public void store(ContextND<ContainerT.Dims> context, in Tile<ContainerT> value) {};

    public typealias SPDims = Size<ContainerT.Dims>;
}
