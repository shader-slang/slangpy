
struct Particle {
    float3 position;
    float angle;
}

uint wang_hash(uint seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}


void init_random_particle(inout Particle particle, float2 min, float2 max) {
    particle.position = float3(rand(), rand(), rand());
    particle.angle = rand();
}