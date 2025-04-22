// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// example.slang
import "slangpy";

// clang-format off
static const float inf = 1.#INF;
// clang-format on

/**
 * Initializes the distance field texture for the Eikonal equation solver.
 * This function analyzes the input image to identify borders between regions
 * and sets up initial distance values and states for each pixel.
 *
 * The function classifies pixels into three states:
 * - Border pixels (state = 3.0): Points exactly on the boundary
 * - Inside pixels (state = 2.0): Points inside the shape with initial distance
 * -inf
 * - Outside pixels (state = 1.0): Points outside the shape with initial
 * distance +inf
 *
 * @param coord The pixel coordinates being processed
 * @param input The input texture containing the source image
 * @param dist_tex The output texture storing distance and state information
 */
void init_eikonal(uint2 coord, Texture2D<float4> input,
                  RWTexture2D<float2> dist_tex) {
  float4 color = input[coord];
  float intensity = (color.r + color.g + color.b) / 3.0;
  float width, height;
  dist_tex.GetDimensions(width, height);

  bool is_border = false;
  bool is_inside = intensity > 0.5;

  // Check if we're on a border by looking at neighbors
  for (int dx = -1; dx <= 1 && !is_border; dx++) {
    for (int dy = -1; dy <= 1 && !is_border; dy++) {
      if (dx == 0 && dy == 0)
        continue;

      int2 neighbor = int2(coord) + int2(dx, dy);
      if (neighbor.x < 0 || neighbor.x >= width || neighbor.y < 0 ||
          neighbor.y >= height)
        continue;

      float4 neighbor_color = input[neighbor];
      float neighbor_intensity =
          (neighbor_color.r + neighbor_color.g + neighbor_color.b) / 3.0;

      if ((neighbor_intensity > 0.5) != is_inside) {
        is_border = true;
      }
    }
  }

  float initial_dist;
  float state;

  if (is_border) {
    initial_dist = 0.0;
    state = 3.0; // border
  } else if (is_inside) {
    initial_dist = -inf;
    state = 2.0; // inside
  } else {
    initial_dist = inf;
    state = 1.0; // outside
  }

  dist_tex[coord] = float2(initial_dist, state);
}

/**
 * Solves the 2D Eikonal equation using the fast sweeping method.
 * This function implements the core mathematical solution for computing
 * the distance field, choosing between 1D and 2D solutions based on the
 * difference between horizontal and vertical distances.
 *
 * @param horizontal The minimum distance value in the horizontal direction
 * @param vertical The minimum distance value in the vertical direction
 * @return The solved distance value
 */
float solve_eikonal_2d(float horizontal, float vertical) {
  if (abs(horizontal - vertical) < 1.0) {
    float sum = horizontal + vertical;
    float dist =
        sum * sum - 2.0 * (horizontal * horizontal + vertical * vertical - 1.0);
    return 0.5 * (sum + sqrt(dist));
  } else {
    return min(horizontal, vertical) + 1.0;
  }
}

/**
 * Main solver function for the Eikonal equation that processes each pixel
 * in the distance field. This function implements one iteration of the
 * fast sweeping method, updating distance values based on neighboring pixels.
 *
 * The function:
 * 1. Skips border points (state = 3.0)
 * 2. Computes minimum distances from horizontal and vertical neighbors
 * 3. Solves the Eikonal equation for the current point
 * 4. Updates the distance while preserving the state
 *
 * @param coord The pixel coordinates being processed
 * @param dist_tex The distance field texture being updated
 */
void solve_eikonal(uint2 coord, RWTexture2D<float2> dist_tex) {
  float2 current = dist_tex[coord];
  float width, height;
  dist_tex.GetDimensions(width, height);

  // Skip border points
  if (current.y == 3.0)
    return;

  float sign = current.x < 0 ? -1.0 : 1.0;
  float current_dist = current.x * sign;

  // Find minimum horizontal neighbors
  float horizontal_min = 1e6;
  if (coord.x > 0) {
    float2 left = dist_tex[uint2(coord.x - 1, coord.y)];
    horizontal_min = min(horizontal_min, sign * left.x);
  }
  if (coord.x < width - 1) {
    float2 right = dist_tex[uint2(coord.x + 1, coord.y)];
    horizontal_min = min(horizontal_min, sign * right.x);
  }

  // Find minimum vertical neighbors
  float vertical_min = 1e6;
  if (coord.y > 0) {
    float2 up = dist_tex[uint2(coord.x, coord.y - 1)];
    vertical_min = min(vertical_min, sign * up.x);
  }
  if (coord.y < height - 1) {
    float2 down = dist_tex[uint2(coord.x, coord.y + 1)];
    vertical_min = min(vertical_min, sign * down.x);
  }

  // Solve eikonal equation
  float eikonal = solve_eikonal_2d(horizontal_min, vertical_min);
  float new_dist = sign * min(current_dist, eikonal);

  dist_tex[coord] = float2(new_dist, current.y);
}

/**
 * Helper function that creates a smooth transition between values.
 * Used for generating smooth isolines in the visualization.
 *
 * @param e0 Start of first transition
 * @param e1 End of first transition
 * @param e2 Start of second transition
 * @param e3 End of second transition
 * @param x Input value
 * @return Smoothly interpolated value
 */
float smooth_pulse(float e0, float e1, float e2, float e3, float x) {
  return smoothstep(e0, e1, x) * (1.0 - smoothstep(e2, e3, x));
}

/**
 * Generates the final visualization with isolines from the distance field.
 * This function creates a visual representation of the distance field with:
 * - Grayscale base representing the distance field
 * - Green isolines at the border (distance = 0)
 * - Red isolines for positive distances (outside)
 * - Blue isolines for negative distances (inside)
 *
 * @param dist_value The distance and state value for the current pixel
 * @return RGBA color for the visualization
 */
float4 generate_isolines(float2 dist_value) {
  float distance = dist_value.x;

  // Base color (grayscale distance field)
  float normalized_dist = distance / 20.0 + 0.5;
  float3 base_color = float3(normalized_dist, normalized_dist, normalized_dist);
  float4 color = float4(base_color, 1.0);

  // Border isoline (green)
  float border_intensity = smooth_pulse(-0.5, -0.1, 0.1, 0.5, distance);
  color.rgb = lerp(color.rgb, float3(0.0, 1.0, 0.0), border_intensity * 0.7);

  // Outside isolines (red)
  if (distance > 0) {
    float spacing = 5.0;
    float phase = distance / spacing;
    float wrapped = fmod(phase, 1.0);
    float outside_intensity = smooth_pulse(0.0, 0.15, 0.35, 0.5, wrapped);
    color.rgb = lerp(color.rgb, float3(1.0, 0.0, 0.0), outside_intensity * 0.5);
  }

  // Inside isolines (blue)
  if (distance < 0) {
    float spacing = 5.0;
    float phase = abs(distance) / spacing;
    float wrapped = fmod(phase, 1.0);
    float inside_intensity = smooth_pulse(0.0, 0.15, 0.35, 0.5, wrapped);
    color.rgb = lerp(color.rgb, float3(0.0, 0.0, 1.0), inside_intensity * 0.5);
  }

  return color;
}

