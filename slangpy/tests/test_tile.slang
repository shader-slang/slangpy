// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import slangpy;

float4 get_centre_pixel(ITile<float4,2> tile) {
    float4 centre_pixel;
    if (tile.load(int2(0, 0), centre_pixel)) {
        return centre_pixel;
    } else {
        return -1;
    }
}

uint2 get_centre_pixel_uint(ITile<uint2, 2> tile) {
    uint2 centre_pixel;
    if (tile.load(int2(0, 0), centre_pixel)) {
        return centre_pixel;
    } else {
        return -1;
    }
}

uint2 get_offset_pixel_uint(ITile<uint2, 2> tile, int2 offset) {
    uint2 centre_pixel;
    if (tile.load(offset, centre_pixel)) {
        return centre_pixel;
    } else {
        return -1;
    }
}

void clear(inout uint2 value) {
    value = 0xffffffff;
}

//sets offset. value comes from call_id, which we receive as array so it is ordered
//according to pixel.
void set_offset_pixel_uint(IRWTile<uint2, 2> tile, int2 offset, uint[2] value) {
    tile.store(offset, uint2(value[0],value[1]));
}

// Generates a buffer of uint2s, with alternating values in x
float generate_x_checkboard(uint2 callid) {
    uint val = callid.x % 2;
    return val;
}

// Generates a buffer of uint2s, with alternating values in x
float generate_y_checkboard(uint2 callid) {
    uint val = callid.y % 2;
    return val;
}

float blur_x(ITile<float, 2> tile) {
    float sum = 0;
    float count = 0;
    for (int i = -3; i <= 3; i++) {
        float pixel;
        if (tile.load(int2(i, 0), pixel)) {
            sum += pixel.x;
            count += 1;
        }
    }
    return sum / count;
}

float blur_y(ITile<float, 2> tile) {
    float sum = 0;
    float count = 0;
    for (int i = -3; i <= 3; i++) {
        float pixel;
        if (tile.load(int2(0, i), pixel)) {
            sum += pixel.x;
            count += 1;
        }
    }
    return sum / count;
}