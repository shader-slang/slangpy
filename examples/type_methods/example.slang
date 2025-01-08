// SPDX-License-Identifier: Apache-2.0
import "randfloatarg";

struct Particle
{
    float3 position;
    float3 velocity;

    __init(float3 p, float3 v)
    {
        position = p;
        velocity = v;
    }

    [mutating]
    void update(float dt)
    {
        position += velocity * dt;
    }
};

