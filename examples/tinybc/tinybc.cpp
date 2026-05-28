// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "sgl/sgl.h"
#include "sgl/core/platform.h"
#include "sgl/core/bitmap.h"
#include "sgl/core/timer.h"
#include "sgl/device/device.h"
#include "sgl/device/shader.h"
#include "sgl/device/command.h"
#include "sgl/device/shader_cursor.h"
#include "sgl/device/shader_object.h"
#include "sgl/device/kernel.h"
#include "sgl/device/query.h"
#include "sgl/device/agility_sdk.h"
#include "sgl/utils/tev.h"

#include <argparse/argparse.hpp>

SGL_EXPORT_AGILITY_SDK

static const std::filesystem::path EXAMPLE_DIR(SGL_EXAMPLE_DIR);

using namespace sgl;

int main(int argc, const char* argv[])
{
    argparse::ArgumentParser args("tinybc");
    args.add_description("Slang-based BC7 - mode 6 compressor");
    args.add_argument("input_path").required().help("Path to the input texture.");
    args.add_argument("-o", "--output_path").help("Optional path to save the decoded BC7 texture.");
    args.add_argument("-s", "--opt_steps").default_value(100).help("Number of optimization (gradient descene) steps.");
    args.add_argument("-b", "--benchmark")
        .default_value(false)
        .implicit_value(true)
        .help("Run in benchmark mode to measure processing time.");
    args.add_argument("-t", "--tev").default_value(false).implicit_value(true).help("Show images in tev image viewer.");
    args.add_argument("-v", "--verbose").default_value(false).implicit_value(true).help("Enable verbose logging.");

    try {
        // args.parse_args(argc, argv);
        SGL_UNUSED(argc, argv);
        args.parse_args(
            {"tinybc",
             "C:/src/sgl/data/test_images/monalisa.jpg",
             "-o",
             "C:/src/sgl/monalisa_bc7.jpg",
             "-t",
             "-b",
             "-v"}
        );
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        return 1;
    }

    std::string input_path = args.get("input_path");
    std::optional<std::string> output_path = args.present("output_path");
    int opt_steps = args.get<int>("opt_steps");
    bool benchmark = args.get<bool>("benchmark");
    bool tev = args.get<bool>("tev");
    bool verbose = args.get<bool>("verbose");

    sgl::static_init();

    {
        ref<Bitmap> input;

        try {
            input = make_ref<Bitmap>(input_path)
                        ->convert(Bitmap::PixelFormat::rgba, Bitmap::ComponentType::float32, false);
        } catch (const std::exception& e) {
            fmt::println("Failed to load input image from {}:\n{}", input_path, e.what());
            return 1;
        }

        uint32_t w = input->width();
        uint32_t h = input->height();

        ref<Device> device = Device::create({
            .enable_debug_layers = verbose,
            .compiler_options = {.include_paths = {EXAMPLE_DIR}},
        });

        SubresourceData initial_data[1] = {{
            .data = input->data(),
            .size = input->buffer_size(),
            .row_pitch = input->width() * 4 * sizeof(float),
        }};

        // Create input texture
        ref<Texture> input_tex = device->create_texture({
            .format = Format::rgba32_float,
            .width = w,
            .height = h,
            .usage = TextureUsage::shader_resource,
            .data = initial_data,
        });

        // Show input texture in tev
        if (tev)
            tev::show_async(input_tex, "tinybc-input");

        // Create decoded texture
        ref<Texture> decoded_tex = device->create_texture({
            .format = Format::rgba32_float,
            .width = w,
            .height = h,
            .usage = TextureUsage::unordered_access,
        });

        std::string constants = fmt::format(
            "export static const bool USE_ADAM = true;\n"
            "export static const uint OPT_STEPS = {};\n",
            opt_steps
        );
        ref<ShaderProgram> program = device->load_program("tinybc.slang", {"main"}, constants);
        ref<ComputeKernel> kernel = device->create_compute_kernel({.program = program});

        uint32_t num_iters = benchmark ? 1000 : 1;

        Timer t;

        // Setup query pool to measure GPU time
        ref<QueryPool> queries = device->create_query_pool({.type = QueryType::timestamp, .count = num_iters * 2});

        // Compress!
        ref<CommandEncoder> command_encoder = device->create_command_encoder();
        for (uint32_t i = 0; i < num_iters; ++i) {
            command_encoder->write_timestamp(queries, i * 2);
            kernel->dispatch(
                uint3(w, h, 1),
                [&](ShaderCursor cursor)
                {
                    cursor["input_tex"] = input_tex;
                    cursor["decoded_tex"] = decoded_tex;
                    cursor["lr"] = 0.1f;
                    cursor["adam_beta_1"] = 0.9f;
                    cursor["adam_beta_2"] = 0.999f;
                },
                command_encoder
            );
            command_encoder->write_timestamp(queries, i * 2 + 1);
        }
        device->submit_command_buffer(command_encoder->finish());

        device->wait();

        double total_cpu_time_sec = t.elapsed_s();

        std::vector<double> times = queries->get_timestamp_results(0, num_iters * 2);
        double comp_time_sec = 0.0;
        for (uint32_t i = 0; i < num_iters; ++i)
            comp_time_sec += (times[i * 2 + 1] - times[i * 2]);
        comp_time_sec /= num_iters;

        // Calculate and print performance metrics
        if (benchmark) {
            double textures_per_sec = 1.0 / comp_time_sec;
            double giga_texels_per_sec = w * h * textures_per_sec / 1e9;
            fmt::println("Benchmark:");
            fmt::println("- Number of optimization steps: {}", opt_steps);
            fmt::println("- Compression time: {:.4g} ms", comp_time_sec * 1e3);
            fmt::println("- Compression throughput: {:.4g} GTexels/s", giga_texels_per_sec);
            fmt::println("- Total CPU time: {:.4g} s", total_cpu_time_sec);
        }

        // Calculate and print PSNR
        ref<Bitmap> decoded = decoded_tex->to_bitmap();
        double mse = 0.0;
        const float* input_data = input->data_as<float>();
        const float* decoded_data = decoded->data_as<float>();
        for (uint32_t i = 0; i < w * h * 4; ++i)
            mse += (input_data[i] - decoded_data[i]) * (input_data[i] - decoded_data[i]);
        mse /= w * h * 4;
        double psnr = 20.0 * std::log10(1.0 / std::sqrt(mse));
        fmt::println("PSNR: {:.4g}", psnr);

        // Show decoded texture in tev
        if (tev)
            tev::show_async(decoded_tex, "tinybc-decoded");

        // Output decoded texture
        if (output_path) {
            decoded_tex->to_bitmap()
                ->convert(Bitmap::PixelFormat::rgb, Bitmap::ComponentType::uint8, true)
                ->write_async(*output_path);
        }

        device->close();
    }

    sgl::static_shutdown();

    return 0;
}
