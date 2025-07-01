
#ifndef INPUT_TENSOR_COUNT
#define INPUT_TENSOR_COUNT 1
#endif

#define READ_MODE_DIRECT 0
#define READ_MODE_INDIRECT 1

#ifndef READ_MODE
#define READ_MODE READ_MODE_DIRECT
#endif

struct Data {
    StructuredBuffer<float> input[INPUT_TENSOR_COUNT];
    RWStructuredBuffer<float> output;
    uint input_tensor_count;
    StructuredBuffer<uint> index_buffer;
    uint index_count;

    float fetch(int buffer, int index)
    {
        return input[buffer][index];
    }
};

#if READ_MODE == READ_MODE_DIRECT
#define READ(buffer, index) data.input[buffer][index]
#elif READ_MODE == READ_MODE_INDIRECT
#define READ(buffer, index) data.fetch(buffer, index)
#endif

ParameterBlock<Data> data;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint3 tid: SV_DispatchThreadID)
{
    float result = 0.0;
    // for (int i = 0; i < 100; ++i) {
    for (int i = 0; i < data.index_count; ++i) {
        uint buffer = data.index_buffer[i];
        result += READ(buffer, tid.x * 1024 + tid.y);
    }
    data.output[tid.x * 1024 + tid.y] = result;
}
