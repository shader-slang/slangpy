// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Slangtorch implementation of PPISP pipeline.
// Based on github.com/nv-tlabs/ppisp (Apache 2.0).
//
// Uses [CUDAKernel] + DiffTensorView for explicit CUDA parallelism.
// Slangtorch handles forward/backward kernel generation.

#include "extensions.slang"

// ============================================================================
// Parameter helper structs
// ============================================================================

struct VignettingParams : IDifferentiable {
    float2 optical_center;
    vector<float, NUM_VIGNETTING_ALPHA_TERMS> alphas;
};

// ============================================================================
// Activation functions
// ============================================================================

[Differentiable]
float bounded_positive(float raw, float min_value) {
    return min_value + log(1.0 + exp(raw));
}

[Differentiable]
float sigmoid_activation(float raw) {
    return 1.0 / (1.0 + exp(-raw));
}

// ============================================================================
// ZCA pinv blocks for color correction
// ============================================================================

static const float COLOR_PINV_BLOCKS[4][4] = {
    {0.0480542, -0.0043631, -0.0043631, 0.0481283},
    {0.0580570, -0.0179872, -0.0179872, 0.0431061},
    {0.0433336, -0.0180537, -0.0180537, 0.0580500},
    {0.0128369, -0.0034654, -0.0034654, 0.0128158}
};

// ============================================================================
// Parameter loading helpers
// ============================================================================

[Differentiable]
VignettingParams get_vignetting_params(DiffTensorView vignetting_params,
                                       uint camera_idx, uint channel_idx) {
    VignettingParams params;
    params.optical_center = vignetting_params.loadVecUniform<2>(uint3(camera_idx, channel_idx, 0));
    params.alphas = vignetting_params.loadVecUniform<NUM_VIGNETTING_ALPHA_TERMS>(uint3(camera_idx, channel_idx, 2));
    return params;
}

// ============================================================================
// Color correction: compute homography
// ============================================================================

[Differentiable]
float3x3 compute_homography(DiffTensorView color_params, uint frame_idx) {
    // Load 8 latent params as 2 vectorized reads
    float4 latent_lo = color_params.loadVecUniform<4>(uint2(frame_idx, 0));
    float4 latent_hi = color_params.loadVecUniform<4>(uint2(frame_idx, 4));
    float latent[8] = {
        latent_lo.x, latent_lo.y, latent_lo.z, latent_lo.w,
        latent_hi.x, latent_hi.y, latent_hi.z, latent_hi.w
    };

    // Map latent -> real offsets via ZCA 2x2 blocks
    float bd_r = COLOR_PINV_BLOCKS[0][0] * latent[0] + COLOR_PINV_BLOCKS[0][1] * latent[1];
    float bd_g = COLOR_PINV_BLOCKS[0][2] * latent[0] + COLOR_PINV_BLOCKS[0][3] * latent[1];
    float rd_r = COLOR_PINV_BLOCKS[1][0] * latent[2] + COLOR_PINV_BLOCKS[1][1] * latent[3];
    float rd_g = COLOR_PINV_BLOCKS[1][2] * latent[2] + COLOR_PINV_BLOCKS[1][3] * latent[3];
    float gd_r = COLOR_PINV_BLOCKS[2][0] * latent[4] + COLOR_PINV_BLOCKS[2][1] * latent[5];
    float gd_g = COLOR_PINV_BLOCKS[2][2] * latent[4] + COLOR_PINV_BLOCKS[2][3] * latent[5];
    float nd_r = COLOR_PINV_BLOCKS[3][0] * latent[6] + COLOR_PINV_BLOCKS[3][1] * latent[7];
    float nd_g = COLOR_PINV_BLOCKS[3][2] * latent[6] + COLOR_PINV_BLOCKS[3][3] * latent[7];

    float3 t_b = float3(0.0 + bd_r, 0.0 + bd_g, 1.0);
    float3 t_r = float3(1.0 + rd_r, 0.0 + rd_g, 1.0);
    float3 t_g = float3(0.0 + gd_r, 1.0 + gd_g, 1.0);
    float3 t_gray = float3(1.0 / 3.0 + nd_r, 1.0 / 3.0 + nd_g, 1.0);

    float3x3 T = float3x3(
        t_b.x, t_r.x, t_g.x,
        t_b.y, t_r.y, t_g.y,
        t_b.z, t_r.z, t_g.z);

    float3x3 skew_mat = float3x3(
        0.0, -t_gray.z, t_gray.y,
        t_gray.z, 0.0, -t_gray.x,
        -t_gray.y, t_gray.x, 0.0);

    float3x3 M = mul(skew_mat, T);

    float3 mr0 = float3(M[0][0], M[0][1], M[0][2]);
    float3 mr1 = float3(M[1][0], M[1][1], M[1][2]);
    float3 mr2 = float3(M[2][0], M[2][1], M[2][2]);

    float3 lam01 = cross(mr0, mr1);
    float3 lam02 = cross(mr0, mr2);
    float3 lam12 = cross(mr1, mr2);

    float n01 = dot(lam01, lam01);
    float n02 = dot(lam02, lam02);
    float n12 = dot(lam12, lam12);

    float3 lam;
    if (n01 >= n02 && n01 >= n12) lam = lam01;
    else if (n02 >= n12) lam = lam02;
    else lam = lam12;

    float3x3 S_inv = float3x3(-1.0, -1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    float3x3 D = float3x3(lam.x, 0.0, 0.0, 0.0, lam.y, 0.0, 0.0, 0.0, lam.z);
    float3x3 H = mul(mul(T, D), S_inv);

    float s = H[2][2] + 1.0e-10;
    H = H * (1.0 / s);
    return H;
}

// ============================================================================
// Main PPISP kernel
// ============================================================================

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void ppisp(no_diff uint batch_size,
           no_diff uint num_cameras,
           no_diff uint num_frames,
           DiffTensorView exposure_params,    // [num_frames]
           DiffTensorView vignetting_params,  // [num_cameras, 3, 5]
           DiffTensorView color_params,       // [num_frames, 8]
           DiffTensorView crf_params,         // [num_cameras, 3, 4]
           DiffTensorView rgb_in,             // [batch_size, 3]
           DiffTensorView rgb_out,            // [batch_size, 3]
           no_diff TensorView<float2> pixel_coords,
           no_diff TensorView<int16_t> camera_idcs,
           no_diff TensorView<int32_t> frame_idcs,
           no_diff float resolution_w,
           no_diff float resolution_h
) {
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= batch_size) return;

    float3 rgb = rgb_in.loadVecOnce<3>(uint2(tid, 0));
    float2 pixel_coord = no_diff pixel_coords[tid];
    int camera_idx = no_diff camera_idcs[tid];
    int frame_idx = no_diff frame_idcs[tid];
    no_diff bool has_frame = frame_idx > -1;
    no_diff bool has_camera = camera_idx > -1;

    // Stage 1: Exposure
    if (has_frame) {
        float exposure_offset = exposure_params.loadUniform(no_diff uint(frame_idx));
        rgb = rgb * exp2(exposure_offset);
    }

    // Stage 2: Vignetting
    if (has_camera) {
        no_diff float max_res = max(resolution_w, resolution_h);
        float2 uv = float2(
            (pixel_coord.x - resolution_w * 0.5) / max_res,
            (pixel_coord.y - resolution_h * 0.5) / max_res);

        [ForceUnroll]
        for (uint i = 0; i < 3; i++) {
            VignettingParams vig = get_vignetting_params(vignetting_params, no_diff uint(camera_idx), i);
            float2 delta = uv - vig.optical_center;
            float r2 = dot(delta, delta);
            float falloff = 1.0;
            float r2_pow = r2;
            [ForceUnroll]
            for (uint j = 0; j < NUM_VIGNETTING_ALPHA_TERMS; j++) {
                falloff += vig.alphas[j] * r2_pow;
                r2_pow *= r2;
            }
            falloff = clamp(falloff, 0.0, 1.0);
            rgb[i] = rgb[i] * falloff;
        }
    }

    // Stage 3: Color correction
    if (has_frame) {
        float3x3 H = compute_homography(color_params, no_diff uint(frame_idx));
        float intensity = rgb.x + rgb.y + rgb.z;
        float3 rgi = float3(rgb.x, rgb.y, intensity);
        rgi = mul(H, rgi);
        rgi = rgi * (intensity / (rgi.z + 1.0e-5));
        rgb = float3(rgi.x, rgi.y, rgi.z - rgi.x - rgi.y);
    }

    // Stage 4: CRF
    if (has_camera) {
        rgb = clamp(rgb, 0.0, 1.0);
        [ForceUnroll]
        for (uint i = 0; i < 3; i++) {
            float4 crf_raw = crf_params.loadVecUniform<4>(uint3(no_diff uint(camera_idx), i, 0));
            float toe_raw = crf_raw.x;
            float shoulder_raw = crf_raw.y;
            float gamma_raw = crf_raw.z;
            float center_raw = crf_raw.w;

            float toe = bounded_positive(toe_raw, 0.3);
            float shoulder = bounded_positive(shoulder_raw, 0.3);
            float gamma_val = bounded_positive(gamma_raw, 0.1);
            float center = sigmoid_activation(center_raw);

            float lerp_val = toe + center * (shoulder - toe);
            float a = (shoulder * center) / lerp_val;
            float b = 1.0 - a;

            float x = rgb[i];
            if (x <= center) {
                rgb[i] = a * pow(max(x / center, 1.0e-6), toe);
            } else {
                rgb[i] = 1.0 - b * pow(max((1.0 - x) / (1.0 - center), 1.0e-6), shoulder);
            }
            rgb[i] = pow(max(rgb[i], 0.0), gamma_val);
        }
    }

    rgb_out.storeVecOnce(uint2(tid, 0), rgb);
}
