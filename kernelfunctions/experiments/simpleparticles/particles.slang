
struct Particle {
    float2 init_pos;
    float2 init_vel;
    float2 position;
    float2 velocity;
    float age;
    float lifetime;
    float size;
    float3 color;

    __init() {
        init_pos = 0;
        init_vel = 0;
        position = 0;
        velocity = 0;
        age = 0;
        lifetime = 0;
        size = 0;
        color = 0;
    }

    __init(Particle other) {
        init_pos = other.init_pos;
        init_vel = other.init_vel;
        position = other.position;
        velocity = other.velocity;
        age = other.age;
        lifetime = other.lifetime;
        size = other.size;
        color = other.color;
    }

    [mutating]
    void init(float2 pos, float2 vel, float lt, float3 col) {
        init_pos = pos;
        init_vel = vel;
        position = pos;
        velocity = vel;
        age = 0;
        lifetime = lt;
        size = 0;
        color = col;
    }

    [mutating]
    void update(float dt) {
        position += velocity * dt;
        age += dt;

        float t = age / lifetime;
        if (t > 1) {
            position = init_pos;
            velocity = init_vel;
            age = 0;
        } else if (t < 0.5) {
            size = t * 2;
        } else {
            size = 1 - (t - 0.5) * 2;
        }
    }

    [mutating]
    void add_impulse(float2 i) {
        velocity += i;
    }

    [mutating]
    void add_point_gravity(float2 target, float strength) {
        float2 dir = target - position;
        float dist = length(dir);
        strength /= max(dist * dist,1);
        dir = normalize(dir);
        velocity += dir * strength;
    }

    [mutating]
    void add_wind(float strength) {
        float2 dir = float2(sin(position.y/50), sin(position.x/50));
        velocity += dir * strength;
    }
}


//-------------------------------------------------------------------------------
// Rasterizing

static const float2[] verts = { 
    float2(-1, -1), float2(-1, 1), float2(1, 1), 
    float2(-1, -1), float2(1, 1), float2(1,-1) };

StructuredBuffer<Particle> draw_particles;
float2 screen_size;

struct v2f {
    float4 position : SV_Position;
    float2 world_pos : TEXCOORD0;
    int index : TEXCOORD1;
}

[shader("vertex")]
v2f vertex_main(uint vert: SV_VertexID)
{
    let particle = draw_particles[vert / verts.getCount()];
    if (particle.age > particle.lifetime) {
        v2f res;
        res.position = float4(-10, -10, 0.0, 1.0);
        res.world_pos = 0;
        res.index = 0;
        return res;        
    }

    let idx = vert % verts.getCount();
    
    var loc_pos = verts[idx] * 20;

    let world_pos = particle.position + loc_pos;

    let vert_pos = world_pos / screen_size * 2 - 1;

    v2f res;
    res.position = float4(vert_pos, 0.0, 1.0);
    res.world_pos = world_pos;
    res.index = vert;
    return res;
}

[shader("fragment")]
float4 fragment_main(v2f v)
    : SV_Target
{
    let particle = draw_particles[v.index / verts.getCount()];
    let dist = length(v.world_pos - particle.position);
    return float4(particle.color*saturate(1-dist/(particle.size*2)),1);
}
