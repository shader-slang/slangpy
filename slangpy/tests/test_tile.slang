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

