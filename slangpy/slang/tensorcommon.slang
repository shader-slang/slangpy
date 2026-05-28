// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

implementing slangpy;

public int _idx<I : __BuiltinIntegerType, let N : int>(I[N] indices, uint stride[N], uint offset)
{
    int idx = 0;
    [ForceUnroll]
    for (int i = 0; i < N; i++) { idx += indices[i].toUInt() * stride[i]; }
    return idx + offset;
}
public int _idx<I : __BuiltinIntegerType, let N : int>(vector<I, N> indices, uint stride[N], uint offset)
{
    int idx = 0;
    int end = N - 1;
    [ForceUnroll]
    for (int i = 0; i < N; i++) { idx += indices[end - i].toUInt() * stride[i]; }
    return idx + offset;
}

public void _slice<let D : int, let SliceD : int>(int[D - SliceD] idx, uint stride[D], uint offset, out uint new_stride[SliceD], out uint new_offset)
{
    [ForceUnroll]
    for (int i = 0; i < SliceD; ++i)
        new_stride[i] = stride[D - SliceD + i];

    new_offset = offset;
    [ForceUnroll]
    for (int i = 0; i < D - SliceD; ++i)
        new_offset += stride[i] * idx[i];
}


public void _idx_accumulate_helper<let D : int, I : __BuiltinIntegerType>( I value, inout uint dest, inout uint dim, uint stride[D] )
{
    dest += value.toInt() * stride[dim++];
}
public int _idx_each<let D : int, each I : __BuiltinIntegerType>(expand each I idx, uint stride[D], uint offset)
{
    static_assert(countof(I) == D, "Mismatch between index count and tensor dimension");
    uint res = 0;
    uint dim = 0;
    expand _idx_accumulate_helper(each idx, res, dim, stride);
    return res + offset;
}
