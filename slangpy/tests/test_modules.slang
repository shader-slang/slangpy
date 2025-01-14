// SPDX-License-Identifier: Apache-2.0
import "slangpy";

struct Material : IDifferentiable {
    float3 color;
    float3 emission;

    [Differentiable]
    __init() {
        color = float3(1, 1, 1);
        emission = float3(0, 0, 0);
    }
};

struct Particle : IDifferentiable {

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