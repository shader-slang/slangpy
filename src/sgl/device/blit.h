// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/fwd.h"
#include "sgl/device/types.h"

#include "sgl/core/object.h"

#include "sgl/math/vector_types.h"

#include <map>

namespace sgl {

/**
 * \brief Utility class for blitting textures and generating mipmaps.
 *
 * Because compute shaders have limited support for UAV texture formats,
 * this class uses a graphics pipeline to blit a texture to a render target.
 */
class Blitter : public Object {
    SGL_OBJECT(Blitter)
public:
    Blitter(Device* device);
    ~Blitter();

    /**
     * \brief Blit a texture (SRV) to another texture (RTV).
     *
     * Blits the full extent of the source texture to the destination texture.
     *
     * \param command_encoder Command encoder.
     * \param dst View of the destination texture.
     * \param src View of the source texture.
     * \param filter Filtering mode to use.
     */
    void blit(
        CommandEncoder* command_encoder,
        TextureView* dst,
        TextureView* src,
        TextureFilteringMode filter = TextureFilteringMode::linear
    );

    void blit(
        CommandEncoder* command_encoder,
        Texture* dst,
        Texture* src,
        TextureFilteringMode filter = TextureFilteringMode::linear
    );

    /**
     * \brief Generate mipmaps for a texture.
     *
     * Repetatively blits the texture to a smaller mip level until the smallest mip level is reached.
     * The texture needs to have mip levels pre-allocated and have usage flags for SRV and RTV.
     * Supports both 2D and 2D array textures.
     *
     * \param command_encoder Command encoder.
     * \param texture Texture to generate mipmaps for.
     * \param layer Array layer to generate mipmaps for.
     */
    void generate_mips(CommandEncoder* command_encoder, Texture* texture, uint32_t layer = 0);

private:
    enum class TextureDataType {
        float_,
        int_,
    };

    enum class TextureLayout {
        texture_2d,
        texture_2d_array,
    };

    struct ProgramKey {
        TextureLayout src_layout;
        TextureDataType src_type;
        TextureDataType dst_type;

        auto operator<=>(const ProgramKey&) const = default;
    };

    ref<ShaderProgram> get_render_program(ProgramKey key);
    ref<RenderPipeline> get_render_pipeline(ProgramKey key, Format dst_format);
    ref<ShaderProgram> get_compute_program(ProgramKey key, Format dst_format);
    ref<ComputePipeline> get_compute_pipeline(ProgramKey key, Format dst_format);

    Device* m_device;
    ref<Sampler> m_linear_sampler;
    ref<Sampler> m_point_sampler;

    std::map<ProgramKey, ref<ShaderProgram>> m_render_program_cache;
    std::map<std::pair<ProgramKey, Format>, ref<RenderPipeline>> m_render_pipeline_cache;

    std::map<std::pair<ProgramKey, Format>, ref<ShaderProgram>> m_compute_program_cache;
    std::map<std::pair<ProgramKey, Format>, ref<ComputePipeline>> m_compute_pipeline_cache;
};

} // namespace sgl
