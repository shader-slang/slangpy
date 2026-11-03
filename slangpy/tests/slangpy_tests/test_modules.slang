// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import "slangpy";

public struct Material : IDifferentiable, IAtomicAddable {
    public float3 color;
    public float3 emission;

    [Differentiable]
    public __init() {
        color = float3(1, 1, 1);
        emission = float3(0, 0, 0);
    }

    public static void atomicAdd(RWByteAddressBuffer buf, uint addr, This value)
    {
        float3::atomicAdd(buf, addr, value.color); addr += sizeof(float3);
        float3::atomicAdd(buf, addr, value.emission);
    }

    public static void atomicAdd(This* buf, uint element, This value)
    {
        float3::atomicAdd(__getAddress(buf[element].color), 0, value.color);
        float3::atomicAdd(__getAddress(buf[element].emission), 0, value.emission);
    }
};

public struct Particle : IDifferentiable, IAtomicAddable {

    public float2 position;
    public float2 velocity;
    public float size;
    public Material material;

    [Differentiable]
    public __init() {
        position = float2(0, 0);
        velocity = float2(0, 0);
        material = Material();
        size = 0;
    }

    [Differentiable]
    public __init(float2 position, float2 velocity) {
        this.position = position;
        this.velocity = velocity;
        this.material = Material();
        this.size = 0.5;
    }

    [mutating]
    public void reset(float2 position, float2 velocity) {
        this.position = position;
        this.velocity = velocity;
        this.material = Material();
        this.size = 0.5;
    }

    [Differentiable]
    public float2 calc_next_position(float dt) {
        return position + velocity * dt;
    }

    [mutating]
    public void update_position(float dt) {
        position = calc_next_position(dt);
    }

    public static void atomicAdd(RWByteAddressBuffer buf, uint addr, This value)
    {
        float2::atomicAdd(buf, addr, value.position); addr += sizeof(float2);
        float2::atomicAdd(buf, addr, value.velocity); addr += sizeof(float2);
        float::atomicAdd(buf, addr, value.size); addr += sizeof(float);
        Material::atomicAdd(buf, addr, value.material);
    }

    public static void atomicAdd(This* buf, uint element, This value)
    {
        float2::atomicAdd(__getAddress(buf[element].position), 0, value.position);
        float2::atomicAdd(__getAddress(buf[element].velocity), 0, value.velocity);
        float::atomicAdd(__getAddress(buf[element].size), 0, value.size);
        Material::atomicAdd(__getAddress(buf[element].material), 0, value.material);
    }
}

public float2[4] get_particle_quad(Particle p) {
    var v = p.velocity;
    if(length(v) < 0.01) {
        v = float2(0, 1);
    }
    float2 up = normalize(v);
    float2 right = float2(-up.y, up.x);
    float2 center = p.position;
    float size = p.size;
    return {
        center + (right + up) * size,
        center + (-right + up) * size,
        center + (-right - up) * size,
        center + (right - up) * size
    };
}

public float3 add_vectors(float3 a, float3 b) {
    return a + b;
}

[Differentiable]
public float3 polynomial(float3 a, float3 b) {
    return a * a + b + 1;
}
