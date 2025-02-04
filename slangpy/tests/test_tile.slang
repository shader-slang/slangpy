// SPDX-License-Identifier: Apache-2.0
import slangpy;

float4 get_centre_pixel(Texture2DTile<float4> tile) {
    return tile.load(int2(0, 0));
}

