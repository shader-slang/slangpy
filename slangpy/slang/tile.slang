// SPDX-License-Identifier: Apache-2.0
implementing slangpy;

public struct Texture2DTile<TexelType : ITexelElement> {
    public Texture2D<TexelType> texture;
    public int2 center;

    public TexelType load(int2 texel_offset) {
        return texture.Load(int3(center + texel_offset,0));
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

