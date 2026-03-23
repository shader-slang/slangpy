extern static const bool ENABLE_FLOAT = false;

struct Test
{
    Conditional<int, !ENABLE_FLOAT> i;
    Conditional<float, ENABLE_FLOAT> f;
}

[shader("compute")]
[numthreads(16, 1, 1)]
void compute_main(uint tid : SV_DispatchThreadID, RWStructuredBuffer<Test> result)
{
    if (ENABLE_FLOAT) {
        result[tid].f = float(tid);
    } else {
        result[tid].i = int(tid);
    }
}
