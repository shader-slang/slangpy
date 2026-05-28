// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This struct holds the BC interpolation weights for a 4x4 texel tile
struct BCWeights : IDifferentiable {
    float w[16];

    __init(float _w)
    {
        for (int i = 0; i < 16; i++)
            w[i] = _w;
    }

    __init(float _w[16])
    {
        for (int i = 0; i < 16; i++)
            w[i] = _w[i];
    }

    [mutating]
    void clamp()
    {
        for (int i = 0; i < 16; i++)
            w[i] = clamp(w[i], 0, 1);
    }

    [mutating]
    void quantize()
    {
        // 3b for the 1st weight
        w[0] = round(w[0] * 7) / 7;
        // and 4b for the others
        for (int i = 1; i < 15; i++)
            w[i] = round(w[i] * 15) / 15;
    }

    BCWeights rsqrt()
    {
        BCWeights r;
        for (int i = 0; i < 16; i++)
            r.w[i] = rsqrt(w[i]);
        return r;
    }
}

BCWeights operator +(float a, BCWeights b)
{
    BCWeights r;
    for (int i = 0; i < 16; i++)
        r.w[i] = a + b.w[i];
    return r;
}

BCWeights operator +(BCWeights a, BCWeights b)
{
    BCWeights r;
    for (int i = 0; i < 16; i++)
        r.w[i] = a.w[i] + b.w[i];
    return r;
}

BCWeights operator -(BCWeights a, BCWeights b)
{
    BCWeights r;
    for (int i = 0; i < 16; i++)
        r.w[i] = a.w[i] - b.w[i];
    return r;
}

BCWeights operator*(BCWeights a, BCWeights b)
{
    BCWeights r;
    for (int i = 0; i < 16; i++)
        r.w[i] = a.w[i] * b.w[i];
    return r;
}

BCWeights operator*(float a, BCWeights b)
{
    BCWeights r;
    for (int i = 0; i < 16; i++)
        r.w[i] = a * b.w[i];
    return r;
}

// This struct holds all of the data for a 4x4 texel tile
struct BCTile : IDifferentiable {
    BCWeights weights;
    float4 min_end_points;
    float4 max_end_points;

    __init()
    {
        weights = BCWeights(0.f);
        min_end_points = 0.f;
        max_end_points = 0.f;
    }

    __init(BCTile.Differential tile)
    {
        weights = BCWeights(tile.weights.w);
        min_end_points = tile.min_end_points;
        max_end_points = tile.max_end_points;
    }

    __init(float _weight, float4 _min_end_points, float4 _max_end_points)
    {
        weights = BCWeights(_weight);
        min_end_points = _min_end_points;
        max_end_points = _max_end_points;
    }

    [mutating]
    void clamp()
    {
        weights.clamp();
        min_end_points = clamp(min_end_points, 0, 1);
        max_end_points = clamp(max_end_points, 0, 1);
    }

    [mutating]
    void quantize()
    {
        weights.quantize();
        min_end_points = round(min_end_points * 127) / 127;
        max_end_points = round(max_end_points * 127) / 127;
    }

    BCTile rsqrt()
    {
        BCTile r;
        r.weights = weights.rsqrt();
        r.min_end_points = rsqrt(min_end_points);
        r.max_end_points = rsqrt(max_end_points);
        return r;
    }

    [BackwardDifferentiable]
    float4 decode(int i) { return (1 - weights.w[i]) * min_end_points + weights.w[i] * max_end_points; }

    // Use 6b interpolation to match API specs
    float4 decode_exact(int i)
    {
        let b = uint(round(64.f * saturate(weights.w[i])));
        let a = 64u - b;
        uint4 u_min_end_points = uint4(127.f * min_end_points) << 1u;
        uint4 u_max_end_points = uint4(127.f * max_end_points) << 1u;
        return (32u + (a * u_min_end_points + b * u_max_end_points) >> 6) / 255.f;
    }
}

BCTile operator*(BCTile a, BCTile b)
{
    BCTile r;
    r.weights = a.weights * b.weights;
    r.min_end_points = a.min_end_points * b.min_end_points;
    r.max_end_points = a.max_end_points * b.max_end_points;
    return r;
}

BCTile operator*(float a, BCTile b)
{
    BCTile r;
    r.weights = a * b.weights;
    r.min_end_points = a * b.min_end_points;
    r.max_end_points = a * b.max_end_points;
    return r;
}

BCTile operator +(float a, BCTile b)
{
    BCTile r;
    r.weights = a + b.weights;
    r.min_end_points = a + b.min_end_points;
    r.max_end_points = a + b.max_end_points;
    return r;
}

BCTile operator +(BCTile a, BCTile b)
{
    BCTile r;
    r.weights = a.weights + b.weights;
    r.min_end_points = a.min_end_points + b.min_end_points;
    r.max_end_points = a.max_end_points + b.max_end_points;
    return r;
}

BCTile operator -(BCTile a, BCTile b)
{
    BCTile r;
    r.weights = a.weights - b.weights;
    r.min_end_points = a.min_end_points - b.min_end_points;
    r.max_end_points = a.max_end_points - b.max_end_points;
    return r;
}
