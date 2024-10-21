
void copy_value(float4 value, out float4 dest) {
    dest = value;
}

void copy_pixel_texture_1d(uint1 position, Texture1D<float4> src, RWTexture1D<float4> dst)
{
    dst[position] = src[position];
}

void copy_pixel_texture_2d(uint2 position, Texture2D<float4> src, RWTexture2D<float4> dst)
{
    dst[position] = src[position];
}

void copy_pixel_texture_1d_array(uint2 position, Texture1DArray<float4> src, RWTexture1DArray<float4> dst)
{
    dst[position.yx] = src[position.yx];
}

void copy_pixel_texture_2d_array(uint3 position, Texture2DArray<float4> src, RWTexture2DArray<float4> dst)
{
    dst[position.yzx] = src[position.yzx];
}

void copy_pixel_texture_3d(uint3 position, Texture3D<float4> src, RWTexture3D<float4> dst)
{
    dst[position] = src[position];
}

void copy_pixel_texture_cube(uint3 position, TextureCube<float4> src, RWTexture3D<float4> dst)
{
}
