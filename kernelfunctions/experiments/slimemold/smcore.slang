
struct Particle {
    float2 position;
    float angle;
}

void init_particle(float2 position, float angle, out Particle particle) {
    particle.position = position;
    particle.angle = angle;
}

void write_particle_pixels(Particle particle, RWTexture2D<float4> output) {
    int2 pixel = int2(particle.position);
    output[pixel] = float4(1.0, 1.0, 1.0, 1.0);
}