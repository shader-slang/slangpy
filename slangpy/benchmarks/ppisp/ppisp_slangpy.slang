// SlangPy implementation of PPISP pipeline.
// Based on github.com/nv-tlabs/ppisp (Apache 2.0).
//
// Pipeline stages:
//   1. Exposure compensation  (Debevec & Malik, SIGGRAPH 1997)
//   2. Vignetting correction  (Goldman, TPAMI 2010)
//   3. Color correction       (Finlayson et al., TPAMI 2019)
//   4. Camera response (CRF)  (Grossberg & Nayar, TPAMI 2004)
//
// Uses ITensor interfaces for SlangPy automatic vectorization.
// Tensor access via subscript operator: tensor[i, j, k]

import slangpy;

// ============================================================================
// Parameter helper structs
// ============================================================================

struct VignettingParams : IDifferentiable {
    float2 optical_center;
    float alphas[NUM_VIGNETTING_ALPHA_TERMS];
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
// ZCA pinv blocks for color correction (constant data)
// ============================================================================

static const float COLOR_PINV_BLOCKS[4][4] = {
    {0.0480542, -0.0043631, -0.0043631, 0.0481283},  // Blue
    {0.0580570, -0.0179872, -0.0179872, 0.0431061},  // Red
    {0.0433336, -0.0180537, -0.0180537, 0.0580500},  // Green
    {0.0128369, -0.0034654, -0.0034654, 0.0128158}   // Neutral
};

// ============================================================================
// Parameter loading helpers
// ============================================================================

[Differentiable]
VignettingParams get_vignetting_params(IDiffTensor<float, 3> vignetting_params,
                                       int camera_idx, int channel_idx) {
    VignettingParams params;
    params.optical_center = float2(
        vignetting_params[camera_idx, channel_idx, 0],
        vignetting_params[camera_idx, channel_idx, 1]);
    [ForceUnroll]
    for (int i = 0; i < NUM_VIGNETTING_ALPHA_TERMS; i++) {
        params.alphas[i] = vignetting_params[camera_idx, channel_idx, 2 + i];
    }
    return params;
}

// ============================================================================
// Color correction: compute homography from latent params
// ============================================================================

[Differentiable]
float3x3 compute_homography(IDiffTensor<float, 2> color_params, int frame_idx) {
    // Load 8 latent params
    float latent[8];
    [ForceUnroll]
    for (int i = 0; i < 8; i++) {
        latent[i] = color_params[frame_idx, i];
    }

    // Map latent -> real offsets via ZCA 2x2 blocks
    // Blue
    float bd_r = COLOR_PINV_BLOCKS[0][0] * latent[0] + COLOR_PINV_BLOCKS[0][1] * latent[1];
    float bd_g = COLOR_PINV_BLOCKS[0][2] * latent[0] + COLOR_PINV_BLOCKS[0][3] * latent[1];
    // Red
    float rd_r = COLOR_PINV_BLOCKS[1][0] * latent[2] + COLOR_PINV_BLOCKS[1][1] * latent[3];
    float rd_g = COLOR_PINV_BLOCKS[1][2] * latent[2] + COLOR_PINV_BLOCKS[1][3] * latent[3];
    // Green
    float gd_r = COLOR_PINV_BLOCKS[2][0] * latent[4] + COLOR_PINV_BLOCKS[2][1] * latent[5];
    float gd_g = COLOR_PINV_BLOCKS[2][2] * latent[4] + COLOR_PINV_BLOCKS[2][3] * latent[5];
    // Neutral
    float nd_r = COLOR_PINV_BLOCKS[3][0] * latent[6] + COLOR_PINV_BLOCKS[3][1] * latent[7];
    float nd_g = COLOR_PINV_BLOCKS[3][2] * latent[6] + COLOR_PINV_BLOCKS[3][3] * latent[7];

    // Target chromaticities = source + offset
    float3 t_b = float3(0.0 + bd_r, 0.0 + bd_g, 1.0);
    float3 t_r = float3(1.0 + rd_r, 0.0 + rd_g, 1.0);
    float3 t_g = float3(0.0 + gd_r, 1.0 + gd_g, 1.0);
    float3 t_gray = float3(1.0 / 3.0 + nd_r, 1.0 / 3.0 + nd_g, 1.0);

    // T = [t_b, t_r, t_g] as columns (row-major: row i = [t_b[i], t_r[i], t_g[i]])
    float3x3 T = float3x3(
        t_b.x, t_r.x, t_g.x,
        t_b.y, t_r.y, t_g.y,
        t_b.z, t_r.z, t_g.z);

    // Skew-symmetric of t_gray
    float3x3 skew_mat = float3x3(
        0.0, -t_gray.z, t_gray.y,
        t_gray.z, 0.0, -t_gray.x,
        -t_gray.y, t_gray.x, 0.0);

    // M = skew * T
    float3x3 M = mul(skew_mat, T);

    // Nullspace via cross product of rows
    float3 mr0 = float3(M[0][0], M[0][1], M[0][2]);
    float3 mr1 = float3(M[1][0], M[1][1], M[1][2]);
    float3 mr2 = float3(M[2][0], M[2][1], M[2][2]);

    float3 lam01 = cross(mr0, mr1);
    float3 lam02 = cross(mr0, mr2);
    float3 lam12 = cross(mr1, mr2);

    float n01 = dot(lam01, lam01);
    float n02 = dot(lam02, lam02);
    float n12 = dot(lam12, lam12);

    // Select cross product with largest magnitude
    float3 lam;
    if (n01 >= n02 && n01 >= n12) lam = lam01;
    else if (n02 >= n12) lam = lam02;
    else lam = lam12;

    // S_inv = [[-1,-1,1],[1,0,0],[0,1,0]]
    float3x3 S_inv = float3x3(-1.0, -1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    // D = diag(lam)
    float3x3 D = float3x3(lam.x, 0.0, 0.0, 0.0, lam.y, 0.0, 0.0, 0.0, lam.z);

    // H = T * D * S_inv
    float3x3 H = mul(mul(T, D), S_inv);

    // Normalize so H[2][2] ~ 1
    float s = H[2][2] + 1.0e-10;
    H = H * (1.0 / s);

    return H;
}

// ============================================================================
// Main PPISP function - per-pixel, SlangPy auto-vectorizes
// ============================================================================

[Differentiable]
float3 ppisp(
    uint batch_size,
    uint num_cameras,
    uint num_frames,
    IDiffTensor<float, 1> exposure_params,    // [num_frames]
    IDiffTensor<float, 3> vignetting_params,  // [num_cameras, 3, 5]
    IDiffTensor<float, 2> color_params,       // [num_frames, 8]
    IDiffTensor<float, 3> crf_params,         // [num_cameras, 3, 4]
    float3 rgb_pixel,
    no_diff float2 pixel_coord,
    int16_t camera_idx,
    int frame_idx,
    no_diff float resolution_w,
    no_diff float resolution_h
) {
    bool has_frame = frame_idx != -1;
    bool has_camera = camera_idx != -1;

    float3 rgb = rgb_pixel;

    // Stage 1: Exposure compensation
    if (has_frame) {
        float exposure_offset = exposure_params[frame_idx];
        rgb = rgb * exp2(exposure_offset);
    }

    // Stage 2: Vignetting
    if (has_camera) {
        float max_res = max(resolution_w, resolution_h);
        float2 uv = float2(
            (pixel_coord.x - resolution_w * 0.5) / max_res,
            (pixel_coord.y - resolution_h * 0.5) / max_res);

        [ForceUnroll]
        for (uint i = 0; i < 3; i++) {
            VignettingParams vig = get_vignetting_params(vignetting_params, camera_idx, int(i));
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
        float3x3 H = compute_homography(color_params, frame_idx);
        float intensity = rgb.x + rgb.y + rgb.z;
        float3 rgi = float3(rgb.x, rgb.y, intensity);
        rgi = mul(H, rgi);
        rgi = rgi * (intensity / (rgi.z + 1.0e-5));
        rgb = float3(rgi.x, rgi.y, rgi.z - rgi.x - rgi.y);
    }

    // Stage 4: CRF (toe-shoulder model)
    if (has_camera) {
        rgb = clamp(rgb, 0.0, 1.0);
        [ForceUnroll]
        for (uint i = 0; i < 3; i++) {
            float toe_raw = crf_params[camera_idx, int(i), 0];
            float shoulder_raw = crf_params[camera_idx, int(i), 1];
            float gamma_raw = crf_params[camera_idx, int(i), 2];
            float center_raw = crf_params[camera_idx, int(i), 3];

            float toe = bounded_positive(toe_raw, 0.3);
            float shoulder = bounded_positive(shoulder_raw, 0.3);
            float gamma_val = bounded_positive(gamma_raw, 0.1);
            float center = sigmoid_activation(center_raw);

            float lerp_val = toe + center * (shoulder - toe);
            float a = (shoulder * center) / lerp_val;
            float b = 1.0 - a;

            float x = rgb[i];
            float y;
            if (x <= center) {
                y = a * pow(max(x / center, 1.0e-6), toe);
            } else {
                y = 1.0 - b * pow(max((1.0 - x) / (1.0 - center), 1.0e-6), shoulder);
            }
            rgb[i] = pow(max(y, 0.0), gamma_val);
        }
    }

    return rgb;
}
