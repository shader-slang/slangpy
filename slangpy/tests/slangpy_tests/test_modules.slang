// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import "slangpy";

struct Material : IDifferentiable, IAtomicAddable {
    float3 color;
    float3 emission;

    [Differentiable]
    __init() {
        color = float3(1, 1, 1);
        emission = float3(0, 0, 0);
    }

    static void atomicAdd(RWByteAddressBuffer buf, uint addr, This value)
    {
        float3::atomicAdd(buf, addr, value.color); addr += sizeof(float3);
        float3::atomicAdd(buf, addr, value.emission);
    }

    static void atomicAdd(This* buf, uint element, This value)
    {
        float3::atomicAdd(&buf[element].color, 0, value.color);
        float3::atomicAdd(&buf[element].emission, 0, value.emission);
    }
};

struct Particle : IDifferentiable, IAtomicAddable {

    float2 position;
    float2 velocity;
    float size;
    Material material;

    [Differentiable]
    __init() {
        position = float2(0, 0);
        velocity = float2(0, 0);
        material = Material();
        size = 0;
    }

    [Differentiable]
    __init(float2 position, float2 velocity) {
        this.position = position;
        this.velocity = velocity;
        this.material = Material();
        this.size = 0.5;
    }

    [mutating]
    void reset(float2 position, float2 velocity) {
        this.position = position;
        this.velocity = velocity;
        this.material = Material();
        this.size = 0.5;
    }

    [Differentiable]
    float2 calc_next_position(float dt) {
        return position + velocity * dt;
    }

    [mutating]
    void update_position(float dt) {
        position = calc_next_position(dt);
    }

    static void atomicAdd(RWByteAddressBuffer buf, uint addr, This value)
    {
        float2::atomicAdd(buf, addr, value.position); addr += sizeof(float2);
        float2::atomicAdd(buf, addr, value.velocity); addr += sizeof(float2);
        float::atomicAdd(buf, addr, value.size); addr += sizeof(float);
        Material::atomicAdd(buf, addr, value.material);
    }

    static void atomicAdd(This* buf, uint element, This value)
    {
        float2::atomicAdd(&buf[element].position, 0, value.position);
        float2::atomicAdd(&buf[element].velocity, 0, value.velocity);
        float::atomicAdd(&buf[element].size, 0, value.size);
        Material::atomicAdd(&buf[element].material, 0, value.material);
    }
}

float2[4] get_particle_quad(Particle p) {
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

float3 add_vectors(float3 a, float3 b) {
    return a + b;
}

[Differentiable]
float3 polynomial(float3 a, float3 b) {
    return a * a + b + 1;
}
