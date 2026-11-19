// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import "slangpy";

public float4 return_value(float4 value) {
    return value;
}

public void copy_value(float4 value, out float4 dest) {
    dest = value;
}

public void copy_pixel_texture_1d(int1 position, Texture1D<float4> src, RWTexture1D<float4> dst)
{
    dst[position] = src[position];
}

public void copy_pixel_texture_2d(int2 position, Texture2D<float4> src, RWTexture2D<float4> dst)
{
    dst[position] = src[position];
}

public void copy_pixel_texture_1d_array(int2 position, Texture1DArray<float4> src, RWTexture1DArray<float4> dst)
{
    dst[position.yx] = src[position.yx];
}

public void copy_pixel_texture_2d_array(int3 position, Texture2DArray<float4> src, RWTexture2DArray<float4> dst)
{
    dst[position.yzx] = src[position.yzx];
}

public void copy_pixel_texture_3d(int3 position, Texture3D<float4> src, RWTexture3D<float4> dst)
{
    dst[position] = src[position];
}

public void copy_pixel_texture_cube(int3 position, TextureCube<float4> src, RWTexture3D<float4> dst)
{
}

public T passthru<T>(T t)
{
    return t;
}

public float sample_texture_1d_broadcast(Texture1D<float> tex)
{
    return tex[uint(2)];
}

public float sample_texture_2d_broadcast(Texture2D<float> tex)
{
    return tex[uint2(3, 1)];
}

public float sample_texture_3d_broadcast(Texture3D<float> tex)
{
    return tex[uint3(2, 1, 3)];
}

public float sample_texture_3d_with_scalar(float value, Texture3D<float> tex)
{
    return tex[uint3(2, 1, 3)] * value;
}
