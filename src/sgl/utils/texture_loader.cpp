// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "texture_loader.h"

#include "sgl/device/device.h"
#include "sgl/device/command.h"
#include "sgl/device/blit.h"
#include "sgl/device/native_formats.h"

#include "sgl/core/error.h"
#include "sgl/core/bitmap.h"
#include "sgl/core/config.h"
#include "sgl/core/dds_file.h"
#include "sgl/core/downsample.h"
#include "sgl/core/file_stream.h"
#include "sgl/core/logger.h"
#include "sgl/core/maths.h"
#include "sgl/core/memory_stream.h"
#include "sgl/core/platform.h"
#include "sgl/core/timer.h"
#include "sgl/core/thread.h"

#include "sgl/stl/bit.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <thread>
#include <utility>

namespace sgl {


// TEMP: stage timing instrumentation (not for commit).
struct LoaderTiming {
    std::atomic<uint64_t> estimate{0}, decode[4]{}, decode_count[4]{}, convert{0}, stage{0}, pool{0}, wait{0},
        create{0}, upload{0}, mips{0}, submit{0};
    static uint64_t now() { return uint64_t(std::chrono::steady_clock::now().time_since_epoch().count()); }
    void reset()
    {
        for (auto* a : {&estimate, &convert, &stage, &pool, &wait, &create, &upload, &mips, &submit})
            *a = 0;
        for (int i = 0; i < 4; ++i)
            decode[i] = decode_count[i] = 0;
    }
    void report(uint64_t wall)
    {
        if (!platform::get_environment_variable("SGL_TEXTURE_LOADER_TIMING"))
            return;
        auto ms = [](uint64_t ns)
        {
            return double(ns) * 1e-6;
        };
        log_info(
            "TIMING wall {:.1f} ms | estimate {:.1f} | decode cpu jpg {:.1f} ({}) png {:.1f} ({}) exr {:.1f} ({}) "
            "other {:.1f} ({}) | convert cpu {:.1f} | stage cpu {:.1f} | main: pool {:.1f} wait {:.1f} create {:.1f} "
            "upload {:.1f} mips {:.1f} "
            "submit+wait {:.1f}",
            ms(wall),
            ms(estimate),
            ms(decode[0]),
            decode_count[0].load(),
            ms(decode[1]),
            decode_count[1].load(),
            ms(decode[2]),
            decode_count[2].load(),
            ms(decode[3]),
            decode_count[3].load(),
            ms(convert),
            ms(stage),
            ms(pool),
            ms(wait),
            ms(create),
            ms(upload),
            ms(mips),
            ms(submit)
        );
    }
};
static LoaderTiming g_timing;
struct ScopedTime {
    std::atomic<uint64_t>& acc;
    uint64_t start{LoaderTiming::now()};
    ~ScopedTime() { acc += LoaderTiming::now() - start; }
};

/// Upload memory that decode workers copy converted images into, so the thread recording uploads only records
/// buffer-to-texture copies instead of copying pixel data itself.
///
/// Pages are persistently mapped upload buffers created by the recording thread. Workers bump-allocate under a
/// mutex. While no page has room, a worker waits and reports this through the starved callback; the recording
/// thread then submits recorded copies or waits for the GPU and recycles pages, which wakes the worker. A page is
/// reused once every allocation from it was recorded and the submissions containing those copies have finished.
/// Images larger than a page, and all images after abort(), stay in CPU memory and are uploaded by the recording
/// thread with CommandEncoder::upload_texture_data instead.
class StagingPool {
public:
    /// Smallest page size in bytes.
    static constexpr size_t MIN_PAGE_SIZE = 64ull * 1024 * 1024;
    /// Largest page size in bytes; larger images are uploaded from CPU memory.
    static constexpr size_t MAX_PAGE_SIZE = 512ull * 1024 * 1024;
    /// Largest total capacity in bytes, unless one page is larger. Creating upload memory costs about 0.2 s per GB;
    /// with workers waiting for recycled pages, 128 MB and 256 MB performed alike, and 512 MB was slower.
    static constexpr size_t MAX_CAPACITY = 128ull * 1024 * 1024;

    struct Allocation {
        uint32_t page;
        size_t offset;
        uint8_t* data;
    };

    /// Create a pool sized for a load.
    /// @param total_bytes Staging bytes of all images in the load, which caps the capacity.
    /// @param max_bytes Staging bytes of the largest image, which sets the page size.
    StagingPool(Device* device, size_t total_bytes, size_t max_bytes)
        : m_device(device)
        , m_recording_thread(std::this_thread::get_id())
    {
        ScopedTime t{g_timing.pool};
        constexpr size_t granularity = 1024 * 1024;
        m_page_size = std::clamp(align_to(granularity, max_bytes), MIN_PAGE_SIZE, MAX_PAGE_SIZE);
        // Small loads need no more than their total, plus slack for allocation alignment.
        m_page_size = std::min(m_page_size, align_to(granularity, total_bytes + total_bytes / 16 + 65536));
        const size_t capacity = std::min(total_bytes, std::max(MAX_CAPACITY, m_page_size));
        const size_t page_count = std::max(size_t(1), div_round_up(capacity, m_page_size));
        m_pages.resize(page_count);
        for (Page& page : m_pages) {
            page.buffer = device->create_buffer({
                .size = m_page_size,
                .memory_type = MemoryType::upload,
                .usage = BufferUsage::copy_source,
                .label = "TextureLoader staging",
            });
            page.data = page.buffer->map<uint8_t>();
        }
    }

    ~StagingPool()
    {
        ScopedTime t{g_timing.pool};
        for (Page& page : m_pages) {
            page.buffer->unmap();
            page.buffer = nullptr;
        }
    }

    StagingPool(const StagingPool&) = delete;
    StagingPool& operator=(const StagingPool&) = delete;

    Device* device() const { return m_device; }

    /// Set the function called, without locks held, when a worker starts waiting for staging memory.
    void set_on_starved(std::function<void()> on_starved)
    {
        std::lock_guard lock(m_mutex);
        m_on_starved = std::move(on_starved);
    }

    /// Allocate staging memory, waiting while all pages are in use. Thread-safe.
    /// Returns nothing if the allocation can never fit a page or the pool was aborted.
    std::optional<Allocation> allocate(size_t size, size_t alignment)
    {
        if (size + alignment > m_page_size)
            return {};
        std::unique_lock lock(m_mutex);
        while (true) {
            if (std::optional<Allocation> allocation = try_allocate(size, alignment))
                return allocation;
            // Only the recording thread frees staging memory, so it must not wait for it. This happens if a wait
            // on that thread runs a decode task; the image is then uploaded from CPU memory.
            if (m_aborted || std::this_thread::get_id() == m_recording_thread)
                return {};
            const uint64_t generation = m_generation;
            m_waiting++;
            std::function<void()> on_starved = m_on_starved;
            lock.unlock();
            if (on_starved)
                on_starved();
            lock.lock();
            m_condition.wait(
                lock,
                [&]
                {
                    return m_generation != generation || m_aborted;
                }
            );
            m_waiting--;
        }
    }

    /// True if a worker waits for staging memory.
    bool waiting()
    {
        std::lock_guard lock(m_mutex);
        return m_waiting > 0;
    }

    /// Stop waiting allocations and make later ones fail, so workers finish after an error.
    void abort()
    {
        {
            std::lock_guard lock(m_mutex);
            m_aborted = true;
        }
        m_condition.notify_all();
    }

    Buffer* buffer(const Allocation& allocation) const { return m_pages[allocation.page].buffer; }

    /// Mark an allocation as recorded into the current, not yet submitted, command encoder.
    void recorded(const Allocation& allocation)
    {
        std::lock_guard lock(m_mutex);
        Page& page = m_pages[allocation.page];
        SGL_ASSERT(page.outstanding > 0);
        page.outstanding--;
        page.unsubmitted = true;
    }

    /// Associate recorded allocations with the submission that contains them.
    void submitted(uint64_t submit_id)
    {
        std::lock_guard lock(m_mutex);
        for (Page& page : m_pages) {
            if (page.unsubmitted) {
                page.last_submit = submit_id;
                page.unsubmitted = false;
            }
        }
    }

    /// Reset pages whose allocations were all recorded and whose submissions finished, and wake waiting workers.
    /// Returns true if a page was reset.
    bool recycle()
    {
        bool reset = false;
        {
            std::lock_guard lock(m_mutex);
            for (Page& page : m_pages) {
                if (page.offset == 0 || page.outstanding > 0 || page.unsubmitted)
                    continue;
                if (page.last_submit != 0 && !m_device->is_submit_finished(page.last_submit))
                    continue;
                page.offset = 0;
                page.last_submit = 0;
                reset = true;
            }
            if (reset)
                m_generation++;
        }
        if (reset)
            m_condition.notify_all();
        return reset;
    }

private:
    std::optional<Allocation> try_allocate(size_t size, size_t alignment)
    {
        const uint32_t page_count = narrow_cast<uint32_t>(m_pages.size());
        for (uint32_t k = 0; k < page_count; ++k) {
            const uint32_t index = (m_current + k) % page_count;
            Page& page = m_pages[index];
            const size_t offset = align_to(alignment, page.offset);
            if (offset + size > m_page_size)
                continue;
            page.offset = offset + size;
            page.outstanding++;
            m_current = index;
            return Allocation{index, offset, page.data + offset};
        }
        return {};
    }

    struct Page {
        ref<Buffer> buffer;
        uint8_t* data{nullptr};
        /// Offset of the first free byte.
        size_t offset{0};
        /// Allocations not yet recorded.
        uint32_t outstanding{0};
        /// True if recorded allocations are not yet submitted.
        bool unsubmitted{false};
        /// Submission containing the last recorded allocation.
        uint64_t last_submit{0};
    };

    Device* m_device;
    std::thread::id m_recording_thread;
    size_t m_page_size{0};
    std::mutex m_mutex;
    std::condition_variable m_condition;
    std::vector<Page> m_pages;
    uint32_t m_current{0};
    /// Incremented when pages are recycled; waiting workers retry when it changes.
    uint64_t m_generation{0};
    uint32_t m_waiting{0};
    bool m_aborted{false};
    std::function<void()> m_on_starved;
};

/// Image data copied into staging memory, ready for a buffer-to-texture copy.
struct StagedImage {
    StagingPool* pool{nullptr};
    StagingPool::Allocation allocation;
    size_t size{0};
    size_t row_pitch{0};
    uint32_t width{0};
    uint32_t height{0};
};

/// Holds a source image to be uploaded to a texture.
/// Data is either a bitmap, a DDS file, or a bitmap already copied to staging memory.
struct SourceImage {
    ref<Bitmap> bitmap;
    ref<DDSFile> dds_file;
    std::optional<StagedImage> staged;
    Format format{Format::undefined};
};

/// Copy a converted bitmap into staging memory and release the bitmap.
/// May wait for staging memory. Leaves the image unchanged if it cannot be staged.
inline void stage_source_image(StagingPool& pool, SourceImage& source_image)
{
    if (!source_image.bitmap)
        return;
    ScopedTime t{g_timing.stage};
    const Bitmap* bitmap = source_image.bitmap;
    const size_t bytes_per_pixel = bitmap->bytes_per_pixel();
    const size_t row_size = bitmap->width() * bytes_per_pixel;
    // Row alignment is backend and format specific. Offsets must be multiples of 512 bytes on D3D12 and of the
    // texel size elsewhere.
    rhi::Size row_alignment = 1;
    if (SLANG_FAILED(pool.device()->rhi_device()->getTextureRowAlignment(
            static_cast<rhi::Format>(source_image.format),
            &row_alignment
        )))
        return;
    const size_t row_pitch = align_to(size_t(row_alignment), row_size);
    const size_t size = row_pitch * bitmap->height();
    const size_t offset_alignment = std::lcm(size_t(512), bytes_per_pixel);
    std::optional<StagingPool::Allocation> allocation = pool.allocate(size, offset_alignment);
    if (!allocation)
        return;

    const uint8_t* src = bitmap->uint8_data();
    if (row_pitch == row_size) {
        std::memcpy(allocation->data, src, size);
    } else {
        for (uint32_t y = 0; y < bitmap->height(); ++y)
            std::memcpy(allocation->data + y * row_pitch, src + y * row_size, row_size);
    }
    source_image.staged = StagedImage{
        .pool = &pool,
        .allocation = *allocation,
        .size = size,
        .row_pitch = row_pitch,
        .width = bitmap->width(),
        .height = bitmap->height(),
    };
    source_image.bitmap = nullptr;
}

inline ref<Bitmap> convert_ya_to_rg(const Bitmap* ya_bitmap)
{
    SGL_ASSERT(ya_bitmap->pixel_format() == Bitmap::PixelFormat::ya);

    ref<Bitmap> rg_bitmap = make_ref<Bitmap>(
        Bitmap::PixelFormat::rg,
        ya_bitmap->component_type(),
        ya_bitmap->width(),
        ya_bitmap->height()
    );
    rg_bitmap->set_srgb_gamma(ya_bitmap->srgb_gamma());
    std::memcpy(rg_bitmap->data(), ya_bitmap->data(), ya_bitmap->buffer_size());

    return rg_bitmap;
}

inline TextureUsage get_effective_texture_usage(Device* device, const TextureLoader::Options& options)
{
    TextureUsage usage = options.usage;
    if (options.generate_mips) {
        usage |= device->has_feature(Feature::rasterization) ? TextureUsage::render_target
                                                             : TextureUsage::unordered_access;
    }
    return usage;
}

inline FormatSupport get_required_format_support(TextureUsage usage)
{
    FormatSupport support = FormatSupport::texture;
    if (is_set(usage, TextureUsage::render_target))
        support |= FormatSupport::render_target;
    if (is_set(usage, TextureUsage::unordered_access))
        support |= FormatSupport::shader_uav_store;
    return support;
}

inline bool has_format_support(Device* device, Format format, FormatSupport required_support)
{
    return (device->get_format_support(format) & required_support) == required_support;
}

/**
 * \brief Determine the texture format given a bitmap.
 *
 * Uses the following option flags to affect the format
 * determination:
 * - \c Options::extend_alpha
 *   RGB bitmap that has no supported format, or needs renderable usage,
 * will be determined as RGBA
 *   (if a RGBA format exists).
 * - \c Options::srgb_mode selects whether 8-bit RGBA uses \c Format::rgba8_unorm_srgb.
 * Automatic interpretation follows bitmap metadata; explicit linear/sRGB overrides it.
 *
 * - \c Options::load_as_normalized
 *   8/16-bit integer bitmap will be determined as normalized resource format.
 *
 * \param info Pixel format, component type and sRGB gamma of the bitmap to determine the format for.
 * \param options Texture loading options.
 * \return A pair containing the determined format and flag if the bitmap needs to be converted to RGBA to match the format.
 */
inline std::pair<Format, bool>
determine_texture_format(Device* device, const Bitmap::Info& info, const TextureLoader::Options& options)
{
    using PixelFormat = Bitmap::PixelFormat;
    using ComponentType = Bitmap::ComponentType;

    enum class FormatFlags {
        none = 0,
        normalized = 1,
        srgb = 2,
    };

    auto make_key = [](PixelFormat pixel_format,
                       ComponentType component_type,
                       FormatFlags flags = FormatFlags::none) constexpr -> uint32_t
    {
        static_assert(Bitmap::PIXEL_FORMAT_COUNT <= 8);
        static_assert(DataStruct::TYPE_COUNT <= 16);
        uint32_t key = 0;
        key |= (1 << uint32_t(pixel_format));
        key |= (1 << uint32_t(component_type)) << 8;
        key |= uint32_t(flags) << 24;
        return key;
    };

    static const std::map<uint32_t, Format> FORMAT_TABLE{
        // PixelFormat::r
        {make_key(PixelFormat::r, ComponentType::int8), Format::r8_sint},
        {make_key(PixelFormat::r, ComponentType::int8, FormatFlags::normalized), Format::r8_snorm},
        {make_key(PixelFormat::r, ComponentType::int16), Format::r16_sint},
        {make_key(PixelFormat::r, ComponentType::int16, FormatFlags::normalized), Format::r16_snorm},
        {make_key(PixelFormat::r, ComponentType::int32), Format::r32_sint},
        {make_key(PixelFormat::r, ComponentType::uint8), Format::r8_uint},
        {make_key(PixelFormat::r, ComponentType::uint8, FormatFlags::normalized), Format::r8_unorm},
        {make_key(PixelFormat::r, ComponentType::uint16), Format::r16_uint},
        {make_key(PixelFormat::r, ComponentType::uint16, FormatFlags::normalized), Format::r16_unorm},
        {make_key(PixelFormat::r, ComponentType::uint32), Format::r32_uint},
        {make_key(PixelFormat::r, ComponentType::float16), Format::r16_float},
        {make_key(PixelFormat::r, ComponentType::float32), Format::r32_float},
        // PixelFormat::rg,
        {make_key(PixelFormat::rg, ComponentType::int8), Format::rg8_sint},
        {make_key(PixelFormat::rg, ComponentType::int8, FormatFlags::normalized), Format::rg8_snorm},
        {make_key(PixelFormat::rg, ComponentType::int16), Format::rg16_sint},
        {make_key(PixelFormat::rg, ComponentType::int16, FormatFlags::normalized), Format::rg16_snorm},
        {make_key(PixelFormat::rg, ComponentType::int32), Format::rg32_sint},
        {make_key(PixelFormat::rg, ComponentType::uint8), Format::rg8_uint},
        {make_key(PixelFormat::rg, ComponentType::uint8, FormatFlags::normalized), Format::rg8_unorm},
        {make_key(PixelFormat::rg, ComponentType::uint16), Format::rg16_uint},
        {make_key(PixelFormat::rg, ComponentType::uint16, FormatFlags::normalized), Format::rg16_unorm},
        {make_key(PixelFormat::rg, ComponentType::uint32), Format::rg32_uint},
        {make_key(PixelFormat::rg, ComponentType::float16), Format::rg16_float},
        {make_key(PixelFormat::rg, ComponentType::float32), Format::rg32_float},
        // PixelFormat::rgb,
        {make_key(PixelFormat::rgb, ComponentType::int32), Format::rgb32_sint},
        {make_key(PixelFormat::rgb, ComponentType::uint32), Format::rgb32_uint},
        {make_key(PixelFormat::rgb, ComponentType::float32), Format::rgb32_float},
        // PixelFormat::rgba,
        {make_key(PixelFormat::rgba, ComponentType::int8), Format::rgba8_sint},
        {make_key(PixelFormat::rgba, ComponentType::int8, FormatFlags::normalized), Format::rgba8_snorm},
        {make_key(PixelFormat::rgba, ComponentType::int16), Format::rgba16_sint},
        {make_key(PixelFormat::rgba, ComponentType::int16, FormatFlags::normalized), Format::rgba16_snorm},
        {make_key(PixelFormat::rgba, ComponentType::int32), Format::rgba32_sint},
        {make_key(PixelFormat::rgba, ComponentType::uint8), Format::rgba8_uint},
        {make_key(PixelFormat::rgba, ComponentType::uint8, FormatFlags::normalized), Format::rgba8_unorm},
        {make_key(PixelFormat::rgba, ComponentType::uint8, FormatFlags::srgb), Format::rgba8_unorm_srgb},
        {make_key(PixelFormat::rgba, ComponentType::uint16), Format::rgba16_uint},
        {make_key(PixelFormat::rgba, ComponentType::uint16, FormatFlags::normalized), Format::rgba16_unorm},
        {make_key(PixelFormat::rgba, ComponentType::uint32), Format::rgba32_uint},
        {make_key(PixelFormat::rgba, ComponentType::float16), Format::rgba16_float},
        {make_key(PixelFormat::rgba, ComponentType::float32), Format::rgba32_float},
    };

    ComponentType component_type = info.component_type;
    PixelFormat pixel_format = info.pixel_format;
    bool convert_to_rgba = false;
    if (pixel_format == PixelFormat::y) {
        if (options.y_handling == YHandling::expand_to_rgba) {
            pixel_format = PixelFormat::rgba;
            convert_to_rgba = true;
        } else {
            pixel_format = PixelFormat::r;
        }
    }
    FormatFlags format_flags = FormatFlags::none;
    if (options.load_as_normalized && DataStruct::is_integer(component_type))
        format_flags = FormatFlags::normalized;

    // Check if bitmap is RGB and we can convert to RGBA.
    if (options.extend_alpha && pixel_format == PixelFormat::rgb) {
        TextureUsage effective_usage = get_effective_texture_usage(device, options);
        FormatSupport required_support = get_required_format_support(effective_usage);
        bool prefer_rgba = is_set(effective_usage, TextureUsage::render_target)
            || is_set(effective_usage, TextureUsage::unordered_access);

        // Some backends can sample RGB formats but cannot use them for renderable mip generation.
        // When alpha extension is enabled, prefer RGBA for renderable usage to keep loading portable.
        bool rgb_format_supported = false;
        if (auto it = FORMAT_TABLE.find(make_key(PixelFormat::rgb, component_type, format_flags));
            it != FORMAT_TABLE.end() && has_format_support(device, it->second, required_support))
            rgb_format_supported = true;

        bool rgba_format_supported = false;
        if (auto it = FORMAT_TABLE.find(make_key(PixelFormat::rgba, component_type, format_flags));
            it != FORMAT_TABLE.end() && has_format_support(device, it->second, required_support))
            rgba_format_supported = true;

        if ((!rgb_format_supported || prefer_rgba) && rgba_format_supported) {
            convert_to_rgba = true;
            pixel_format = PixelFormat::rgba;
        }
    }

    // Handle YA bitmap based on ya_handling option.
    if (pixel_format == PixelFormat::ya) {
        if (options.ya_handling == YAHandling::expand_to_rgba) {
            pixel_format = PixelFormat::rgba;
            convert_to_rgba = true;
        } else {
            pixel_format = PixelFormat::rg;
        }
    }

    // Use sRGB format if requested and supported.
    const bool srgb
        = options.srgb_mode == SRGBMode::srgb || (options.srgb_mode == SRGBMode::automatic && info.srgb_gamma);
    if (srgb && pixel_format == PixelFormat::rgba && component_type == ComponentType::uint8)
        format_flags = FormatFlags::srgb;

    // Find texture format.
    auto it = FORMAT_TABLE.find(make_key(pixel_format, component_type, format_flags));
    if (it == FORMAT_TABLE.end())
        SGL_THROW("Unsupported bitmap format: {} {}", pixel_format, component_type);

    return {it->second, convert_to_rgba};
}

inline std::pair<Format, bool>
determine_texture_format(Device* device, const Bitmap* bitmap, const TextureLoader::Options& options)
{
    SGL_ASSERT(bitmap != nullptr);
    return determine_texture_format(
        device,
        Bitmap::Info{
            .width = bitmap->width(),
            .height = bitmap->height(),
            .pixel_format = bitmap->pixel_format(),
            .component_type = bitmap->component_type(),
            .channel_count = bitmap->channel_count(),
            .srgb_gamma = bitmap->srgb_gamma(),
        },
        options
    );
}

inline std::pair<TextureType, uint32_t> get_texture_type_and_layer_count(DDSFile::TextureType type, uint32_t array_size)
{
    switch (type) {
    case DDSFile::TextureType::texture_1d:
        if (array_size > 1)
            return {TextureType::texture_1d_array, array_size};
        return {TextureType::texture_1d, 1};
    case DDSFile::TextureType::texture_2d:
        if (array_size > 1)
            return {TextureType::texture_2d_array, array_size};
        return {TextureType::texture_2d, 1};
    case DDSFile::TextureType::texture_3d:
        return {TextureType::texture_3d, 1};
    case DDSFile::TextureType::texture_cube:
        if (array_size > 1)
            return {TextureType::texture_cube_array, array_size * 6};
        return {TextureType::texture_cube, 6};
    default:
        SGL_THROW("Invalid DDS texture type {}", type);
    }
}

/// Halve an 8-bit or 16-bit unorm y/ya/rgb/rgba bitmap with even dimensions using a 2x2 box filter in linear light.
/// Runs single-threaded, since textures are already loaded in parallel. Returns nullptr when the fast path does not
/// apply.
inline ref<Bitmap> halve_unorm_bitmap(const Bitmap& bitmap, bool srgb)
{
    using PF = Bitmap::PixelFormat;
    using CT = Bitmap::ComponentType;
    const PF pixel_format = bitmap.pixel_format();
    const CT component_type = bitmap.component_type();
    // 16-bit sRGB data is not supported by the kernels (and no 16-bit sRGB texture formats exist).
    const bool supported_type = component_type == CT::uint8 || (component_type == CT::uint16 && !srgb);
    if (!supported_type
        || (pixel_format != PF::y && pixel_format != PF::ya && pixel_format != PF::rgb && pixel_format != PF::rgba)
        || bitmap.width() % 2 != 0 || bitmap.height() % 2 != 0)
        return nullptr;

    const uint32_t channels = bitmap.channel_count();
    const bool has_alpha = pixel_format == PF::ya || pixel_format == PF::rgba;
    ref<Bitmap> result = make_ref<Bitmap>(pixel_format, component_type, bitmap.width() / 2, bitmap.height() / 2);
    const size_t src_row_pitch = size_t(bitmap.width()) * bitmap.bytes_per_pixel();
    const size_t dst_row_pitch = size_t(result->width()) * result->bytes_per_pixel();
    if (component_type == CT::uint8) {
        downsample_2x_box_unorm8(
            bitmap.data_as<uint8_t>(),
            src_row_pitch,
            result->data_as<uint8_t>(),
            dst_row_pitch,
            result->width(),
            result->height(),
            channels,
            srgb,
            has_alpha
        );
    } else {
        downsample_2x_box_unorm16(
            bitmap.data_as<uint16_t>(),
            src_row_pitch,
            result->data_as<uint16_t>(),
            dst_row_pitch,
            result->width(),
            result->height(),
            channels
        );
    }
    return result;
}

// Reduce in linear light, quantizing to the texture format after each halving.
inline ref<Bitmap> reduce_bitmap(ref<Bitmap> bitmap, Format format, uint32_t max_mip_count)
{
    if (max_mip_count == 0 || uint32_t(stdx::bit_width(std::max(bitmap->width(), bitmap->height()))) <= max_mip_count)
        return bitmap;

    const bool srgb = get_format_info(format).is_srgb_format();
    const auto pixel_format = bitmap->pixel_format();
    const auto component_type = bitmap->component_type();
    const bool source_gamma = bitmap->srgb_gamma();
    while (uint32_t(stdx::bit_width(std::max(bitmap->width(), bitmap->height()))) > max_mip_count) {
        if (ref<Bitmap> halved = halve_unorm_bitmap(*bitmap, srgb)) {
            bitmap = std::move(halved);
            continue;
        }
        // A borrowed view changes interpretation without mutating the caller's bitmap.
        Bitmap source(pixel_format, component_type, bitmap->width(), bitmap->height(), 0, {}, bitmap->data(), srgb);
        bitmap = source.convert(pixel_format, Bitmap::ComponentType::float32, false);
        const uint32_t width = std::max(1u, bitmap->width() / 2);
        const uint32_t height = std::max(1u, bitmap->height() / 2);
        bitmap = bitmap->resample(width, height);
        bitmap = bitmap->convert(pixel_format, component_type, srgb);
    }
    bitmap->set_srgb_gamma(source_gamma);
    return bitmap;
}

inline SourceImage convert_bitmap(Device* device, ref<Bitmap> bitmap, const TextureLoader::Options& options)
{
    using PixelFormat = Bitmap::PixelFormat;

    auto [format, convert_to_rgba] = determine_texture_format(device, bitmap, options);

    // Reduce before expanding channels: replicated luminance and the opaque alpha that conversion adds reduce to
    // the same values, so this gives identical results while converting fewer pixels.
    bitmap = reduce_bitmap(std::move(bitmap), format, options.max_mip_count);

    if (bitmap->pixel_format() == PixelFormat::ya && options.ya_handling == YAHandling::preserve_as_rg) {
        bitmap = convert_ya_to_rg(bitmap);
    } else if (convert_to_rgba) {
        bitmap = bitmap->convert(PixelFormat::rgba, bitmap->component_type(), bitmap->srgb_gamma());
    }

    return SourceImage{
        .bitmap = std::move(bitmap),
        .format = format,
    };
}

/// Read the image header and return read options that let the decoder produce the pixel layout the texture needs,
/// so no separate conversion pass is required.
inline Bitmap::ReadOptions get_bitmap_read_options(
    Device* device,
    Stream* stream,
    Bitmap::FileFormat file_format,
    const TextureLoader::Options& options
)
{
    using PixelFormat = Bitmap::PixelFormat;
    if (file_format != Bitmap::FileFormat::png && file_format != Bitmap::FileFormat::jpg
        && file_format != Bitmap::FileFormat::exr)
        return {};
    const size_t position = stream->tell();
    const Bitmap::Info info = Bitmap::read_info(stream, file_format);
    stream->seek(position);
    if (info.pixel_format != PixelFormat::y && info.pixel_format != PixelFormat::ya
        && info.pixel_format != PixelFormat::rgb)
        return {};
    const bool convert_to_rgba = determine_texture_format(device, info, options).second;
    return {.expand_to_rgba = convert_to_rgba};
}

inline SourceImage load_source_image(Device* device, Stream* stream, const TextureLoader::Options& options)
{
    SourceImage source_image;
    if (DDSFile::detect_dds_file(stream)) {
        source_image.dds_file = ref(new DDSFile(stream));
        source_image.format = get_format(DXGI_FORMAT(source_image.dds_file->dxgi_format()));
    } else if (auto file_format = Bitmap::detect_file_format(stream); file_format != Bitmap::FileFormat::unknown) {
        int index = file_format == Bitmap::FileFormat::jpg ? 0
            : file_format == Bitmap::FileFormat::png       ? 1
            : file_format == Bitmap::FileFormat::exr       ? 2
                                                           : 3;
        ScopedTime t{g_timing.decode[index]};
        g_timing.decode_count[index]++;
        const Bitmap::ReadOptions read_options = get_bitmap_read_options(device, stream, file_format, options);
        source_image.bitmap = ref(new Bitmap(stream, file_format, read_options));
    }
    return source_image;
}

inline Format apply_dds_srgb_mode(Format format, SRGBMode mode)
{
    if (mode == SRGBMode::automatic)
        return format;

    // Reinterpret the stored texels without changing their encoding or layout.
    static constexpr std::pair<Format, Format> FORMAT_PAIRS[]{
        {Format::rgba8_unorm, Format::rgba8_unorm_srgb},
        {Format::bgra8_unorm, Format::bgra8_unorm_srgb},
        {Format::bgrx8_unorm, Format::bgrx8_unorm_srgb},
        {Format::bc1_unorm, Format::bc1_unorm_srgb},
        {Format::bc2_unorm, Format::bc2_unorm_srgb},
        {Format::bc3_unorm, Format::bc3_unorm_srgb},
        {Format::bc7_unorm, Format::bc7_unorm_srgb},
    };
    for (const auto& [linear, srgb] : FORMAT_PAIRS) {
        if (format == linear || format == srgb)
            return mode == SRGBMode::srgb ? srgb : linear;
    }
    return format;
}

inline SourceImage
load_source_image(Device* device, const std::filesystem::path& path, const TextureLoader::Options& options)
{
    FileStream stream(path, FileStream::Mode::read);
    return load_source_image(device, &stream, options);
}

inline SourceImage convert_source_image(Device* device, SourceImage source_image, const TextureLoader::Options& options)
{
    ScopedTime t{g_timing.convert};
    if (source_image.bitmap)
        return convert_bitmap(device, std::move(source_image.bitmap), options);
    if (source_image.dds_file)
        source_image.format = apply_dds_srgb_mode(source_image.format, options.srgb_mode);
    return source_image;
}

inline SourceImage load_and_convert_source_image(Device* device, Stream* stream, const TextureLoader::Options& options)
{
    return convert_source_image(device, load_source_image(device, stream, options), options);
}

inline SourceImage
load_and_convert_source_image(Device* device, const std::filesystem::path& path, const TextureLoader::Options& options)
{
    // Decode and close the file before converting. Conversion may wait on nested pool work (resampling), and the
    // waiting worker runs other queued loads meanwhile; holding the file open would nest one open handle per queued
    // load and can exhaust the open file limit for large batches.
    return convert_source_image(device, load_source_image(device, path, options), options);
}

inline SourceImage load_and_convert_source_image(
    Device* device,
    const TextureLoader::EncodedImageView& image,
    const TextureLoader::Options& options
)
{
    try {
        MemoryStream stream(image.data.data(), image.data.size());
        return load_and_convert_source_image(device, &stream, options);
    } catch (const std::exception& e) {
        SGL_THROW("Failed to decode image '{}': {}", image.name.empty() ? "<memory>" : image.name, e.what());
    }
}

inline ref<Texture> create_texture(
    Device* device,
    Blitter* blitter,
    CommandEncoder* command_encoder,
    SourceImage source_image,
    const TextureLoader::Options& options
)
{
    if (source_image.staged) {
        const StagedImage& staged = *source_image.staged;
        bool allocate_mips = options.allocate_mips || options.generate_mips;

        uint64_t t0 = LoaderTiming::now();
        ref<Texture> texture = device->create_texture({
            .type = TextureType::texture_2d,
            .format = source_image.format,
            .width = staged.width,
            .height = staged.height,
            .mip_count = allocate_mips ? ALL_MIPS : 1u,
            .usage = get_effective_texture_usage(device, options),
        });
        g_timing.create += LoaderTiming::now() - t0;
        {
            ScopedTime t{g_timing.upload};
            command_encoder->copy_buffer_to_texture(
                texture,
                0,
                0,
                uint3(0),
                staged.pool->buffer(staged.allocation),
                staged.allocation.offset,
                staged.size,
                staged.row_pitch
            );
            staged.pool->recorded(staged.allocation);
        }
        if (options.generate_mips) {
            ScopedTime t{g_timing.mips};
            blitter->generate_mips(command_encoder, texture);
        }

        return texture;
    } else if (source_image.bitmap) {
        const Bitmap* bitmap = source_image.bitmap;
        bool allocate_mips = options.allocate_mips || options.generate_mips;

        TextureUsage usage = get_effective_texture_usage(device, options);

        uint64_t t0 = LoaderTiming::now();
        ref<Texture> texture = device->create_texture({
            .type = TextureType::texture_2d,
            .format = source_image.format,
            .width = bitmap->width(),
            .height = bitmap->height(),
            .mip_count = allocate_mips ? ALL_MIPS : 1u,
            .usage = usage,
        });
        g_timing.create += LoaderTiming::now() - t0;
        SubresourceData subresource_data{
            .data = bitmap->data(),
            .row_pitch = bitmap->width() * bitmap->bytes_per_pixel(),
        };

        {
            ScopedTime t{g_timing.upload};
            command_encoder->upload_texture_data(texture, 0, 0, subresource_data);
        }
        if (options.generate_mips) {
            ScopedTime t{g_timing.mips};
            blitter->generate_mips(command_encoder, texture);
        }

        return texture;
    } else if (source_image.dds_file) {
        const DDSFile* dds_file = source_image.dds_file;
        const auto& [texture_type, layer_count]
            = get_texture_type_and_layer_count(dds_file->type(), dds_file->array_size());
        uint32_t first_mip = 0;
        uint32_t width = dds_file->width();
        uint32_t height = dds_file->height();
        uint32_t depth = dds_file->depth();
        while (options.max_mip_count != 0
               && uint32_t(stdx::bit_width(std::max({width, height, depth}))) > options.max_mip_count
               && first_mip + 1 < dds_file->mip_count()) {
            ++first_mip;
            width = std::max(1u, width / 2);
            height = std::max(1u, height / 2);
            depth = std::max(1u, depth / 2);
        }
        if (options.max_mip_count != 0
            && uint32_t(stdx::bit_width(std::max({width, height, depth}))) > options.max_mip_count) {
            log_warn(
                "DDS has no mip fitting max_mip_count {}; using smallest available mip {}x{}x{}",
                options.max_mip_count,
                width,
                height,
                depth
            );
        }
        short_vector<SubresourceData, 16> subresource_data;
        for (uint32_t layer_index = 0; layer_index < layer_count; ++layer_index) {
            for (uint32_t mip_index = first_mip; mip_index < dds_file->mip_count(); ++mip_index) {
                uint32_t row_pitch;
                uint32_t slice_pitch;
                dds_file->get_subresource_pitch(mip_index, &row_pitch, &slice_pitch);
                uint32_t mip_depth = dds_file->type() == DDSFile::TextureType::texture_3d
                    ? std::max(1u, dds_file->depth() >> mip_index)
                    : 1u;
                subresource_data.push_back({
                    .data = dds_file->get_subresource_data(mip_index, layer_index),
                    .size = size_t(slice_pitch) * mip_depth,
                    .row_pitch = row_pitch,
                    .slice_pitch = slice_pitch,
                });
            }
        }

        return device->create_texture({
            .type = texture_type,
            .format = source_image.format,
            .width = width,
            .height = height,
            .depth = depth,
            .array_length = dds_file->array_size(),
            .mip_count = dds_file->mip_count() - first_mip,
            .usage = options.usage,
            .data = subresource_data,
        });
    } else {
        SGL_THROW("Unsupported source image type");
    }
}

// Estimate temporary CPU memory per texture to choose a batch size, not to predict exact process peak memory.
// Take the largest decode/conversion working set, then conservatively add output and upload staging storage.
// File and encoded inputs include decoding and their source bitmap; caller-owned bitmap and encoded-byte storage,
// and final GPU textures are excluded. Format expansion, codec scratch and upload overhead are estimates rather than
// backend-specific allocation accounting.
/// Estimated resources for loading one source image.
struct SourceEstimate {
    /// Temporary CPU memory in bytes, used to choose batch sizes.
    size_t memory{0};
    /// Bytes of upload staging memory, including row alignment; zero for sources that are not staged.
    size_t staging{0};
};

inline SourceEstimate estimate_texture_load_memory(
    const Bitmap::Info& info,
    const TextureLoader::Options& options,
    size_t decode_peak_bytes,
    bool owns_bitmap
)
{
    uint32_t channels = info.channel_count;
    if (info.pixel_format == Bitmap::PixelFormat::rgb
        || (info.pixel_format == Bitmap::PixelFormat::y && options.y_handling == YHandling::expand_to_rgba)
        || (info.pixel_format == Bitmap::PixelFormat::ya && options.ya_handling == YAHandling::expand_to_rgba))
        channels = 4;
    const size_t component_bytes = DataStruct::type_size(info.component_type);
    const size_t pixel_bytes = channels * component_bytes;
    uint32_t width = info.width, height = info.height;
    // Decoding finishes before conversion; each conversion temporarily retains its input and output.
    size_t current_bytes = owns_bitmap ? size_t(width) * height * info.channel_count * component_bytes : 0;
    size_t peak = std::max(decode_peak_bytes, current_bytes);
    if (channels != info.channel_count || info.pixel_format == Bitmap::PixelFormat::ya) {
        const size_t converted_bytes = size_t(width) * height * pixel_bytes;
        peak = std::max(peak, current_bytes + converted_bytes);
        current_bytes = converted_bytes;
    }
    const uint32_t mip_count = uint32_t(stdx::bit_width(std::max(width, height)));
    if (options.max_mip_count && mip_count > options.max_mip_count) {
        // Downsampling uses float images plus horizontal-pass scratch. The first halving bounds later peaks.
        const uint32_t next_width = std::max(1u, width / 2), next_height = std::max(1u, height / 2);
        const size_t float_pixel_bytes = size_t(channels) * sizeof(float);
        const size_t linear = size_t(width) * height * float_pixel_bytes;
        const size_t horizontal = size_t(next_width) * height * float_pixel_bytes;
        const size_t reduced = size_t(next_width) * next_height * float_pixel_bytes;
        peak = std::max({peak, current_bytes + linear, linear + horizontal + 2 * reduced});
        const uint32_t reductions = mip_count - options.max_mip_count;
        width = std::max(1u, width >> reductions);
        height = std::max(1u, height >> reductions);
    }
    const size_t output = size_t(width) * height * pixel_bytes;
    // Allow a retained result and a staging copy, including conservative row alignment.
    const size_t staging = align_to(size_t(256), size_t(width) * pixel_bytes) * height;
    // Fixed overhead and 12.5% padding are heuristic allowances for uncounted allocations.
    const size_t estimate = peak + output + staging + 32768;
    return {.memory = estimate + div_round_up(estimate, size_t(8)), .staging = staging};
}

inline SourceEstimate estimate_source_memory(const Bitmap* bitmap, const TextureLoader::Options& options)
{
    return estimate_texture_load_memory(
        {bitmap->width(), bitmap->height(), bitmap->pixel_format(), bitmap->component_type(), bitmap->channel_count()},
        options,
        0,
        false
    );
}

inline SourceEstimate estimate_source_memory(Stream* stream, const TextureLoader::Options& options)
{
    const size_t encoded = stream->size();
    if (DDSFile::detect_dds_file(stream)) {
        // DDSFile copies encoded data from either stream type; allow that copy and upload staging.
        // A borrowed encoded-image source is not counted here.
        const size_t estimate = 2 * encoded + 65536;
        return {.memory = estimate + div_round_up(estimate, size_t(8))};
    }
    const auto format = Bitmap::detect_file_format(stream);
    const auto info = Bitmap::read_info(stream, format);
    const size_t pixels = size_t(info.width) * info.height * info.channel_count;
    const size_t decoded = pixels * DataStruct::type_size(info.component_type);
    // Cover the result plus image-sized codec scratch, including a possible copy of encoded input.
    // A borrowed encoded-image source is not counted here.
    size_t decode_peak_bytes = encoded + 2 * decoded;
#if !SGL_HAS_OPENEXR
    if (format == Bitmap::FileFormat::exr)
        // TinyEXR retains encoded input, channel planes and the interleaved result.
        decode_peak_bytes = encoded + pixels * sizeof(float) + decoded;
#endif
    return estimate_texture_load_memory(info, options, decode_peak_bytes, true);
}

inline SourceEstimate estimate_source_memory(const std::filesystem::path& path, const TextureLoader::Options& options)
{
    FileStream stream(path, FileStream::Mode::read);
    return estimate_source_memory(&stream, options);
}

inline SourceEstimate
estimate_source_memory(const TextureLoader::EncodedImageView& image, const TextureLoader::Options& options)
{
    try {
        MemoryStream stream(image.data.data(), image.data.size());
        return estimate_source_memory(&stream, options);
    } catch (const std::exception& e) {
        SGL_THROW("Failed to inspect image '{}': {}", image.name.empty() ? "<memory>" : image.name, e.what());
    }
}

inline SourceImage
load_and_convert_source_image(Device* device, const Bitmap* bitmap, const TextureLoader::Options& options)
{
    return convert_bitmap(device, ref(const_cast<Bitmap*>(bitmap)), options);
}

/// Decode tasks of one batch, consumed in completion order.
/// Each task queues its batch-local index when it finishes, successfully or not, so the consumer can process
/// whichever image is ready instead of waiting for them in submission order.
class DecodeTasks {
public:
    explicit DecodeTasks(size_t count)
        : m_handles(count)
    {
    }

    DecodeTasks(const DecodeTasks&) = delete;
    DecodeTasks& operator=(const DecodeTasks&) = delete;

    size_t size() const { return m_handles.size(); }

    /// Launch the task for batch-local index `index`.
    template<typename Func>
    void launch(size_t index, Func&& func)
    {
        m_handles[index] = thread::do_async(
            [this, index, func = std::forward<Func>(func)]
            {
                // Queue the index on failure too; wait_next() rethrows the task's exception.
                struct Notify {
                    DecodeTasks* tasks;
                    size_t index;
                    ~Notify() { tasks->notify(index); }
                } notify{this, index};
                func();
            }
        );
    }

    /// Wait for the next finished task, release it and return its batch-local index.
    /// Rethrows the exception of a failed task.
    size_t wait_next()
    {
        size_t index;
        {
            std::unique_lock lock(m_mutex);
            m_condition.wait(
                lock,
                [this]
                {
                    return !m_finished.empty();
                }
            );
            index = m_finished.front();
            m_finished.pop_front();
        }
        thread::task_wait_and_release(std::exchange(m_handles[index], nullptr));
        return index;
    }

    /// Release and return the batch-local index of a finished task, if any.
    /// Rethrows the exception of a failed task.
    std::optional<size_t> try_next()
    {
        size_t index;
        {
            std::lock_guard lock(m_mutex);
            if (m_finished.empty())
                return {};
            index = m_finished.front();
            m_finished.pop_front();
        }
        // The task queued its index as its last action and completes momentarily. Waiting for an incomplete task
        // would let nanothread run other queued tasks on this thread, and a decode task waiting for staging memory
        // that only this thread can free would never return.
        thread::TaskHandle handle = std::exchange(m_handles[index], nullptr);
        while (!thread::task_query(handle))
            std::this_thread::yield();
        thread::task_wait_and_release(handle);
        return index;
    }

    /// Number of wake() calls so far.
    uint64_t wake_count()
    {
        std::lock_guard lock(m_mutex);
        return m_wake_count;
    }

    /// Wake a thread blocked in wait_for_change(). Thread-safe.
    void wake()
    {
        {
            std::lock_guard lock(m_mutex);
            m_wake_count++;
        }
        m_condition.notify_one();
    }

    /// Block until a task has finished or wake() was called since wake_count() returned `observed_wake_count`.
    void wait_for_change(uint64_t observed_wake_count)
    {
        std::unique_lock lock(m_mutex);
        m_condition.wait(
            lock,
            [&]
            {
                return !m_finished.empty() || m_wake_count != observed_wake_count;
            }
        );
    }

    /// Wait for and release all remaining tasks, discarding their exceptions.
    void drain() noexcept
    {
        for (thread::TaskHandle& handle : m_handles) {
            if (handle) {
                try {
                    thread::task_wait_and_release(std::exchange(handle, nullptr));
                } catch (...) {
                }
            }
        }
    }

private:
    void notify(size_t index)
    {
        {
            std::lock_guard lock(m_mutex);
            m_finished.push_back(index);
        }
        m_condition.notify_one();
    }

    std::vector<thread::TaskHandle> m_handles;
    std::mutex m_mutex;
    std::condition_variable m_condition;
    std::deque<size_t> m_finished;
    uint64_t m_wake_count{0};
};

/// Records texture uploads and submits them in chunks.
/// Submitting once enough data is staged, and waiting only for submissions older than the last few, lets the GPU
/// process uploads while the CPU decodes and records more, and bounds the staging memory in use.
class UploadThrottle {
public:
    /// Staged bytes after which recorded uploads are submitted.
    /// The staging memory of MAX_IN_FLIGHT + 1 chunks stays below slang-rhi's default staging retain size
    /// (256 MB), so staging pages are reused instead of newly allocated and first touched; 256 MB chunks were
    /// measured to make recording uploads about 30% slower.
    static constexpr size_t SUBMIT_BYTES = 64ull * 1024 * 1024;
    /// Maximum number of submissions the GPU may still be processing when recording continues.
    static constexpr size_t MAX_IN_FLIGHT = 2;

    /// @param staging_pool Pool whose pages are recycled as submissions finish (optional).
    explicit UploadThrottle(Device* device, StagingPool* staging_pool = nullptr)
        : m_device(device)
        , m_staging_pool(staging_pool)
        , m_command_encoder(device->create_command_encoder())
    {
    }

    CommandEncoder* command_encoder() const { return m_command_encoder; }

    /// Account for uploads recorded since the last call; submits once enough data is staged.
    void add(size_t staged_bytes)
    {
        m_recorded = true;
        m_staged_bytes += staged_bytes;
        if (m_staged_bytes >= SUBMIT_BYTES)
            submit();
        if (m_staging_pool)
            m_staging_pool->recycle();
    }

    /// Make progress towards freeing staging memory: submit recorded uploads, recycle staging pages, or wait for
    /// the oldest submission. Returns false if there is nothing to do.
    bool advance()
    {
        if (m_recorded) {
            submit();
            return true;
        }
        if (m_staging_pool && m_staging_pool->recycle())
            return true;
        if (!m_in_flight.empty()) {
            ScopedTime t{g_timing.submit};
            wait_oldest();
            if (m_staging_pool)
                m_staging_pool->recycle();
            return true;
        }
        return false;
    }

    /// Submit remaining uploads and wait until the GPU has processed all submissions.
    void finish()
    {
        if (m_recorded)
            submit();
        ScopedTime t{g_timing.submit};
        while (!m_in_flight.empty())
            wait_oldest();
    }

private:
    void submit()
    {
        ScopedTime t{g_timing.submit};
        const uint64_t submit_id = m_device->submit_command_buffer(m_command_encoder->finish());
        m_in_flight.push_back(submit_id);
        if (m_staging_pool)
            m_staging_pool->submitted(submit_id);
        m_command_encoder = m_device->create_command_encoder();
        m_staged_bytes = 0;
        m_recorded = false;
        while (m_in_flight.size() > MAX_IN_FLIGHT)
            wait_oldest();
    }

    /// Wait for the oldest submission in flight.
    void wait_oldest()
    {
        // slang-rhi's CUDA queue signals submission fences when work is submitted, not when it completes, so wait
        // for the whole queue there.
        if (m_device->type() == DeviceType::cuda) {
            m_device->wait();
            m_in_flight.clear();
            return;
        }
        m_device->wait_for_submit(m_in_flight.front());
        m_in_flight.pop_front();
    }

    Device* m_device;
    StagingPool* m_staging_pool;
    ref<CommandEncoder> m_command_encoder;
    size_t m_staged_bytes{0};
    bool m_recorded{false};
    std::deque<uint64_t> m_in_flight;
};

/// Bytes staged for uploading a source image.
inline size_t staged_size(const SourceImage& source_image)
{
    if (source_image.staged)
        return source_image.staged->size;
    if (source_image.bitmap)
        return source_image.bitmap->buffer_size();
    if (source_image.dds_file)
        return source_image.dds_file->resource_size();
    return 0;
}

/// Load ordered batches within an estimated memory budget.
/// Source is std::filesystem::path, const Bitmap*, or EncodedImageView; options has one entry per source.
/// Caller-owned bitmaps and encoded bytes must remain alive until this function returns.
/// If use_staging_pool is set, workers copy converted bitmaps into a StagingPool, which is passed to consume.
/// consume(begin, images, tasks, staging_pool) runs on the calling thread after launching each batch's tasks.
/// begin is the batch's offset in sources; images and tasks use batch-local indices.
/// The callback calls tasks.wait_next() once per image and accesses only the image it returns.
/// It consumes/releases the images and finishes their GPU uploads before returning normally.
/// Both are borrowed for the callback only; this helper drains remaining tasks on failure.
template<typename Source, typename Consume>
void load_batches(
    Device* device,
    std::span<const Source> sources,
    std::span<const TextureLoader::Options> options,
    uint64_t memory_budget,
    bool use_staging_pool,
    Consume&& consume
)
{
    // Inspect all sources in parallel before choosing batches; pixel loading still reopens each file.
    std::vector<SourceEstimate> estimates(sources.size());
    g_timing.reset();
    const uint64_t t_wall = LoaderTiming::now();
    {
        ScopedTime t_estimate{g_timing.estimate};
        thread::parallel_for(
            thread::blocked_range<size_t>(0, sources.size()),
            [&](const thread::blocked_range<size_t>& range)
            {
                for (size_t i : range)
                    estimates[i] = estimate_source_memory(sources[i], options[i]);
            }
        );
    }
    struct Report {
        uint64_t start;
        ~Report() { g_timing.report(LoaderTiming::now() - start); }
    } report{t_wall};

    // The pool outlives the decode tasks of every batch, which copy into it.
    std::optional<StagingPool> staging_pool;
    // slang-rhi's CUDA queue signals submission fences when work is submitted, not when the stream completes it,
    // so is_submit_finished() cannot tell when staging pages are free again. CUDA uploads from CPU memory instead.
    if (use_staging_pool && device->type() != DeviceType::cuda) {
        size_t total_staging = 0;
        size_t max_staging = 0;
        for (const SourceEstimate& estimate : estimates) {
            total_staging += estimate.staging;
            max_staging = std::max(max_staging, estimate.staging);
        }
        if (total_staging > 0)
            staging_pool.emplace(device, total_staging, max_staging);
    }

    for (size_t begin = 0; begin < sources.size();) {
        size_t end = begin;
        uint64_t remaining = memory_budget;
        while (end < sources.size() && remaining) {
            const uint64_t estimate = estimates[end].memory;
            if (end > begin && estimate > remaining)
                break;
            remaining -= std::min(estimate, remaining); // An oversized image is admitted alone.
            ++end;
        }
        std::vector<SourceImage> images(end - begin);
        DecodeTasks tasks(images.size());
        if (staging_pool)
            staging_pool->set_on_starved(
                [&tasks]
                {
                    tasks.wake();
                }
            );
        try {
            for (size_t i = 0; i < images.size(); ++i)
                tasks.launch(
                    i,
                    [&, i]
                    {
                        images[i] = load_and_convert_source_image(device, sources[begin + i], options[begin + i]);
                        if (staging_pool)
                            stage_source_image(*staging_pool, images[i]);
                    }
                );
            consume(begin, std::span(images), tasks, staging_pool ? &*staging_pool : nullptr);
        } catch (...) {
            // Workers borrow batch storage; drain them before propagating the original failure. Workers waiting
            // for staging memory would wait forever, so abort the pool first.
            if (staging_pool)
                staging_pool->abort();
            tasks.drain();
            throw;
        }
        if (staging_pool)
            staging_pool->set_on_starved({});
        begin = end;
    }
}

inline std::vector<ref<Texture>> create_textures(
    Device* device,
    Blitter* blitter,
    std::span<SourceImage> source_images,
    DecodeTasks& source_image_tasks,
    std::span<const TextureLoader::Options> options,
    StagingPool* staging_pool
)
{
    SGL_ASSERT(source_images.size() == source_image_tasks.size());
    SGL_ASSERT(source_images.size() == options.size());
    std::vector<ref<Texture>> textures(source_images.size());
    UploadThrottle throttle(device, staging_pool);
    for (size_t n = 0; n < source_images.size();) {
        // Read the wake count first so that a worker starting to wait during the checks below is not missed.
        const uint64_t wake_count = source_image_tasks.wake_count();
        std::optional<size_t> next = source_image_tasks.try_next();
        if (!next) {
            // Workers waiting for staging memory need recorded uploads to be submitted and finished.
            if (staging_pool && staging_pool->waiting() && throttle.advance())
                continue;
            ScopedTime t{g_timing.wait};
            source_image_tasks.wait_for_change(wake_count);
            continue;
        }
        const size_t i = *next;
        ++n;
        const size_t staged_bytes = staged_size(source_images[i]);
        textures[i]
            = create_texture(device, blitter, throttle.command_encoder(), std::move(source_images[i]), options[i]);
        throttle.add(staged_bytes);
    }
    throttle.finish();

    return textures;
}

inline ref<Texture> create_texture_array(
    Device* device,
    Blitter* blitter,
    std::span<SourceImage> source_images,
    DecodeTasks& source_image_tasks,
    const TextureLoader::Options& options,
    ref<Texture> texture,
    uint32_t first_layer,
    uint32_t layer_count
)
{
    SGL_ASSERT(source_images.size() == source_image_tasks.size());
    SGL_ASSERT(source_images.size() > 0);

    bool allocate_mips = options.allocate_mips || options.generate_mips;

    TextureUsage usage = get_effective_texture_usage(device, options);

    UploadThrottle throttle(device);

    for (size_t n = 0; n < source_images.size(); ++n) {
        const size_t i = source_image_tasks.wait_next();
        SourceImage source_image = std::move(source_images[i]);
        const Bitmap* bitmap = source_image.bitmap;
        if (!bitmap)
            SGL_THROW("Texture array requires all source images to be bitmaps");

        if (!texture) {
            texture = device->create_texture({
                .type = TextureType::texture_2d_array,
                .format = source_image.format,
                .width = bitmap->width(),
                .height = bitmap->height(),
                .array_length = layer_count,
                .mip_count = allocate_mips ? ALL_MIPS : 1u,
                .usage = usage,
            });
        } else {
            if (bitmap->width() != texture->width() || bitmap->height() != texture->height()
                || source_image.format != texture->format())
                SGL_THROW("Texture array requires all bitmaps to have the same dimensions and format");
        }

        SubresourceData subresource_data{
            .data = bitmap->data(),
            .size = bitmap->buffer_size(),
            .row_pitch = bitmap->width() * bitmap->bytes_per_pixel(),
        };
        CommandEncoder* command_encoder = throttle.command_encoder();
        command_encoder->upload_texture_data(texture, first_layer + narrow_cast<uint32_t>(i), 0, subresource_data);
        const size_t staged_bytes = bitmap->buffer_size();
        /// Release bitmap to free CPU memory
        source_image.bitmap = nullptr;

        if (options.generate_mips)
            blitter->generate_mips(command_encoder, texture, first_layer + narrow_cast<uint32_t>(i));
        throttle.add(staged_bytes);
    }
    throttle.finish();

    return texture;
}

TextureLoader::TextureLoader(ref<Device> device, uint64_t memory_budget)
    : m_device(std::move(device))
    , m_memory_budget(memory_budget)
{
    SGL_CHECK(memory_budget > 0, "memory_budget must be positive");
    m_blitter = ref(new Blitter(m_device));
}

TextureLoader::~TextureLoader() = default;

ref<Texture> TextureLoader::load_texture(const Bitmap* bitmap, std::optional<Options> options_)
{
    Options options = options_.value_or(Options{});
    SourceImage source_image = convert_bitmap(m_device, ref(const_cast<Bitmap*>(bitmap)), options);
    ref<CommandEncoder> command_encoder = m_device->create_command_encoder();
    ref<Texture> texture = create_texture(m_device, m_blitter, command_encoder, std::move(source_image), options);
    m_device->submit_command_buffer(command_encoder->finish());
    return texture;
}

ref<Texture> TextureLoader::load_texture(Stream* stream, std::optional<Options> options_)
{
    Options options = options_.value_or(Options{});
    SourceImage source_image = load_and_convert_source_image(m_device.get(), stream, options);
    ref<CommandEncoder> command_encoder = m_device->create_command_encoder();
    ref<Texture> texture = create_texture(m_device, m_blitter, command_encoder, std::move(source_image), options);
    m_device->submit_command_buffer(command_encoder->finish());
    return texture;
}

ref<Texture> TextureLoader::load_texture(const std::filesystem::path& path, std::optional<Options> options_)
{
    Options options = options_.value_or(Options{});
    SourceImage source_image = load_and_convert_source_image(m_device.get(), path, options);
    ref<CommandEncoder> command_encoder = m_device->create_command_encoder();
    ref<Texture> texture = create_texture(m_device, m_blitter, command_encoder, std::move(source_image), options);
    m_device->submit_command_buffer(command_encoder->finish());
    return texture;
}

std::vector<ref<Texture>>
TextureLoader::load_textures(std::span<const Bitmap*> bitmaps, std::optional<Options> options_)
{
    std::vector<Options> options(bitmaps.size(), options_.value_or(Options{}));
    return load_textures(bitmaps, options);
}

std::vector<ref<Texture>>
TextureLoader::load_textures(std::span<const Bitmap*> bitmaps, std::span<const Options> options)
{
    SGL_CHECK(bitmaps.size() == options.size(), "Number of options must be equal to the number of bitmaps");

    std::vector<ref<Texture>> textures(bitmaps.size());
    load_batches<const Bitmap*>(
        m_device,
        bitmaps,
        options,
        m_memory_budget,
        true,
        [&](size_t begin, auto images, auto& tasks, StagingPool* staging_pool)
        {
            auto batch = create_textures(
                m_device,
                m_blitter,
                images,
                tasks,
                options.subspan(begin, images.size()),
                staging_pool
            );
            std::move(batch.begin(), batch.end(), textures.begin() + begin);
        }
    );
    return textures;
}

std::vector<ref<Texture>>
TextureLoader::load_textures(std::span<const std::filesystem::path> paths, std::optional<Options> options_)
{
    std::vector<Options> options(paths.size(), options_.value_or(Options{}));
    return load_textures(paths, options);
}

std::vector<ref<Texture>>
TextureLoader::load_textures(std::span<const std::filesystem::path> paths, std::span<const Options> options)
{
    SGL_CHECK(paths.size() == options.size(), "Number of options must be equal to the number of paths");

    std::vector<ref<Texture>> textures(paths.size());
    load_batches<std::filesystem::path>(
        m_device,
        paths,
        options,
        m_memory_budget,
        true,
        [&](size_t begin, auto images, auto& tasks, StagingPool* staging_pool)
        {
            auto batch = create_textures(
                m_device,
                m_blitter,
                images,
                tasks,
                options.subspan(begin, images.size()),
                staging_pool
            );
            std::move(batch.begin(), batch.end(), textures.begin() + begin);
        }
    );
    return textures;
}

std::vector<ref<Texture>>
TextureLoader::load_textures(std::span<const EncodedImageView> images, std::optional<Options> options_)
{
    std::vector<Options> options(images.size(), options_.value_or(Options{}));
    return load_textures(images, options);
}

std::vector<ref<Texture>>
TextureLoader::load_textures(std::span<const EncodedImageView> images, std::span<const Options> options)
{
    SGL_CHECK(images.size() == options.size(), "Number of options must be equal to the number of images");

    std::vector<ref<Texture>> textures(images.size());
    load_batches<EncodedImageView>(
        m_device,
        images,
        options,
        m_memory_budget,
        true,
        [&](size_t begin, auto source_images, auto& tasks, StagingPool* staging_pool)
        {
            auto batch = create_textures(
                m_device,
                m_blitter,
                source_images,
                tasks,
                options.subspan(begin, source_images.size()),
                staging_pool
            );
            std::move(batch.begin(), batch.end(), textures.begin() + begin);
        }
    );
    return textures;
}

ref<Texture> TextureLoader::load_texture_array(std::span<const Bitmap*> bitmaps, std::optional<Options> options_)
{
    if (bitmaps.empty())
        return nullptr;

    Options options = options_.value_or(Options{});

    ref<Texture> texture;
    const std::vector<Options> per_image_options(bitmaps.size(), options);
    load_batches<const Bitmap*>(
        m_device,
        bitmaps,
        per_image_options,
        m_memory_budget,
        false,
        [&](size_t begin, auto images, auto& tasks, StagingPool*)
        {
            texture = create_texture_array(
                m_device,
                m_blitter,
                images,
                tasks,
                options,
                texture,
                narrow_cast<uint32_t>(begin),
                narrow_cast<uint32_t>(bitmaps.size())
            );
        }
    );
    return texture;
}

ref<Texture>
TextureLoader::load_texture_array(std::span<const std::filesystem::path> paths, std::optional<Options> options_)
{
    if (paths.empty())
        return nullptr;

    Options options = options_.value_or(Options{});

    ref<Texture> texture;
    const std::vector<Options> per_image_options(paths.size(), options);
    load_batches<std::filesystem::path>(
        m_device,
        paths,
        per_image_options,
        m_memory_budget,
        false,
        [&](size_t begin, auto images, auto& tasks, StagingPool*)
        {
            texture = create_texture_array(
                m_device,
                m_blitter,
                images,
                tasks,
                options,
                texture,
                narrow_cast<uint32_t>(begin),
                narrow_cast<uint32_t>(paths.size())
            );
        }
    );
    return texture;
}

} // namespace sgl
