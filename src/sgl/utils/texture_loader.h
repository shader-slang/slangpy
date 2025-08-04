// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/fwd.h"
#include "sgl/device/resource.h"

#include "sgl/core/fwd.h"
#include "sgl/core/object.h"

#include <filesystem>

namespace sgl {

/**
 * \brief Utility class for loading textures from bitmaps and image files.
 */
class SGL_API TextureLoader : public sgl::Object {
    SGL_OBJECT(TextureLoader)
public:
    TextureLoader(ref<Device> device);
    ~TextureLoader();

    struct SGL_API Options {
        /// Load 8/16-bit integer data as normalized resource format.
        bool load_as_normalized{true};
        /// Use \c Format::rgba8_unorm_srgb format if bitmap is 8-bit RGBA with sRGB gamma.
        bool load_as_srgb{true};
        /// Extend RGB to RGBA if RGB texture format is not available.
        bool extend_alpha{true};
        /// Allocate mip levels for the texture.
        bool allocate_mips{false};
        /// Generate mip levels for the texture.
        bool generate_mips{false};
        /// Resource usage flags for the texture.
        /// \c TextureUsage::render_target will be added automatically if \c generate_mips is true.
        TextureUsage usage{TextureUsage::shader_resource};
    };

    /**
     * \brief Load a texture from a bitmap.
     *
     * \param bitmap Bitmap to load.
     * \param options Texture loading options.
     * \return New texture object.
     */
    ref<Texture> load_texture(const Bitmap* bitmap, std::optional<Options> options = {});

    /**
     * \brief Load a texture from an image file.
     *
     * \param path Image file path.
     * \param options Texture loading options.
     * \return New texture object.
     */
    ref<Texture> load_texture(const std::filesystem::path& path, std::optional<Options> options = {});

    /**
     * \brief Load textures from a list of bitmaps.
     *
     * \param bitmaps Bitmaps to load.
     * \param options Texture loading options.
     * \return List of new of texture objects.
     */
    std::vector<ref<Texture>> load_textures(std::span<const Bitmap*> bitmaps, std::optional<Options> options = {});

    /**
     * \brief Load textures from a list of image files.
     *
     * \param paths Image file paths.
     * \param options Texture loading options.
     * \return List of new texture objects.
     */
    std::vector<ref<Texture>>
    load_textures(std::span<std::filesystem::path> paths, std::optional<Options> options = {});

    /**
     * \brief Load a texture array from a list of bitmaps.
     *
     * All bitmaps need to have the same format and dimensions.
     *
     * \param bitmaps Bitmaps to load.
     * \param options Texture loading options.
     * \return New texture array object.
     */
    ref<Texture> load_texture_array(std::span<const Bitmap*> bitmaps, std::optional<Options> options = {});

    /**
     * \brief Load a texture array from a list of image files.
     *
     * All images need to have the same format and dimensions.
     *
     * \param paths Image file paths.
     * \param options Texture loading options.
     * \return New texture array object.
     */
    ref<Texture> load_texture_array(std::span<std::filesystem::path> paths, std::optional<Options> options = {});

private:
    ref<Device> m_device;
    ref<Blitter> m_blitter;
};

} // namespace sgl
