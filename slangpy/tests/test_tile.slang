// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import slangpy;

float4 get_centre_pixel(ITile<float4,2> tile) {
    float4 centre_pixel;
    tile.load(int2(0, 0), centre_pixel);
    return centre_pixel;
}

