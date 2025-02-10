// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
implementing slangpy;

public interface ITile<T, let N : int> {
    public void load(vector<int, N> call_id_offset, out T value);
}

public interface IRWTile<T, let N : int> {
    public void load(vector<int, N> call_id_offset, out T value);
    public void store(vector<int, N> call_id_offset, in T value);
}

public struct Tile<ContainerT : IIndexableType> : ITile<ContainerT.IndexType, ContainerT.Dims>, IRWTile<ContainerT.IndexType, ContainerT.Dims> {
    public ContainerT data;
    public ContextND<ContainerT.Dims> context;

    public void load(vector<int, ContainerT.Dims> call_id_offset, out ContainerT.IndexType value) {
        ContextND<ContainerT.Dims> sub_context = context;
        for (int i = 0; i < ContainerT.Dims; i++) {
            sub_context.call_id[i] += call_id_offset[i];
        }
        data.load(sub_context, value);
    }

    public void store(vector<int, ContainerT.Dims> call_id_offset, in ContainerT.IndexType value) {
        ContextND<ContainerT.Dims> sub_context = context;
        for (int i = 0; i < ContainerT.Dims; i++) {
            sub_context.call_id[i] += call_id_offset[i];
        }
        data.store(sub_context, value);
    }

    typealias SPMarshall = TileType<ContainerT>;
    typealias SPDims = Size<ContainerT.Dims>;
}

public struct TileType<ContainerT : IIndexableType>  {
    public ContainerT data;

    public void load(ContextND<ContainerT.Dims> context, out Tile<ContainerT> value) { value = { data, context }; }
    public void store(ContextND<ContainerT.Dims> context, in Tile<ContainerT> value) {};

    typealias SPMarshall = NoneType;
    typealias SPDims = Size<ContainerT.Dims>;
}

/*

public struct Texture2DTile<TexelType : ITexelElement> {
    public Texture2DType<TexelType> data;
    public Context2D context;

    public void load(int2 offset, out TexelType value) {
        Context2D sub_context = context;
        sub_context.call_id[0] += offset.y;
        sub_context.call_id[1] += offset.x;
        data.load(sub_context, value);
    }
}

public struct Texture2DTileType<TexelType : ITexelElement> {
    public Texture2D<TexelType> texture;
    public int2 size;

    public void load(Context2D context, out Texture2DTile<TexelType> value) { value = { texture, context.call_id_vec.yx }; }
    public void store(Context2D context, in Texture2DTile<TexelType> value) {};
}

public struct RWTexture2DTile<TexelType : ITexelElement> {
    public RWTexture2D<TexelType> texture;
    public int2 center;

    public TexelType load(int2 texel_offset) {
        return texture.Load(center + texel_offset);
    }

    public void store(int2 texel_offset, TexelType value) {
        texture.Store(center + texel_offset, value);
    }
}

public struct RWTexture2DTileType<TexelType : ITexelElement> {
    public Texture2D<TexelType> texture;
    public int2 size;

    public void load(Context2D context, out Texture2DTile<TexelType> value) { value = { texture, context.call_id_vec.yx }; }
    public void store(Context2D context, in Texture2DTile<TexelType> value) {};
}

*/
