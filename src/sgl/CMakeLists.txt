
add_library(sgl SHARED)

target_sources(sgl PRIVATE
    sgl.natvis

    sgl.h
    sgl.cpp

    app/app.cpp
    app/app.h

    core/bitmap.cpp
    core/bitmap.h
    core/crypto.cpp
    core/crypto.h
    core/data_struct.cpp
    core/data_struct.h
    core/data_type.h
    core/dds_file.cpp
    core/dds_file.h
    core/enum.h
    core/error.cpp
    core/error.h
    core/file_stream.cpp
    core/file_stream.h
    core/file_system_watcher.cpp
    core/file_system_watcher.h
    core/format.h
    core/fwd.h
    core/hash.h
    core/input.cpp
    core/input.h
    core/logger.cpp
    core/logger.h
    core/macros.h
    core/maths.h
    core/memory_mapped_file_stream.cpp
    core/memory_mapped_file_stream.h
    core/memory_mapped_file.cpp
    core/memory_mapped_file.h
    core/memory_stream.cpp
    core/memory_stream.h
    core/object.cpp
    core/object.h
    core/platform_linux.cpp
    core/platform_macos.cpp
    core/platform_windows.cpp
    core/platform.cpp
    core/platform.h
    core/plugin.cpp
    core/plugin.h
    core/resolver.h
    core/short_vector.h
    core/static_vector.h
    core/stream.h
    core/string.cpp
    core/string.h
    core/thread.cpp
    core/thread.h
    core/timer.cpp
    core/timer.h
    core/traits.h
    core/type_utils.h
    core/window.cpp
    core/window.h

    device/agility_sdk.h
    device/blit.cpp
    device/blit.h
    device/blit.slang
    device/buffer_cursor.cpp
    device/buffer_cursor.h
    device/command.cpp
    device/command.h
    device/coopvec.h
    device/coopvec.cpp
    device/cuda_interop.cpp
    device/cuda_interop.h
    device/cuda_utils.cpp
    device/cuda_utils.h
    device/cursor_utils.cpp
    device/cursor_utils.h
    device/debug_logger.h
    device/device_resource.cpp
    device/device_resource.h
    device/device.cpp
    device/device.h
    device/fence.cpp
    device/fence.h
    device/formats.cpp
    device/formats.h
    device/fwd.h
    device/helpers.h
    device/helpers.cpp
    device/hot_reload.h
    device/hot_reload.cpp
    device/input_layout.cpp
    device/input_layout.h
    device/kernel.cpp
    device/kernel.h
    device/native_formats.h
    device/nvapi.slang
    device/nvapi.slangh
    device/pipeline.cpp
    device/pipeline.h
    device/print.cpp
    device/print.h
    device/print.slang
    device/query.cpp
    device/query.h
    device/raytracing.cpp
    device/raytracing.h
    device/reflection.cpp
    device/reflection.h
    device/resource.cpp
    device/resource.h
    device/sampler.cpp
    device/sampler.h
    device/shader_cursor.cpp
    device/shader_cursor.h
    device/shader_object.cpp
    device/shader_object.h
    device/shader_offset.h
    device/shader.cpp
    device/shader.h
    device/slang_utils.h
    device/surface.cpp
    device/surface.h
    device/types.cpp
    device/types.h

    math/constants.h
    math/constants.slang
    math/float16.cpp
    math/float16.h
    math/matrix_math.h
    math/matrix_types.h
    math/matrix.h
    math/quaternion_math.h
    math/quaternion_types.h
    math/quaternion.h
    math/ray.h
    math/ray.slang
    math/scalar_math.h
    math/scalar_types.h
    math/vector_math.h
    math/vector_types.h
    math/vector.h

    stl/bit.h

    ui/fwd.h
    ui/imgui.slang
    ui/ui.cpp
    ui/ui.h
    ui/widgets.cpp
    ui/widgets.h

    utils/renderdoc.cpp
    utils/renderdoc.h
    utils/slangpy.cpp
    utils/slangpy.h
    utils/tev.cpp
    utils/tev.h
    utils/texture_loader.cpp
    utils/texture_loader.h

    ${SGL_IMGUI_DIR}/imconfig.h
    ${SGL_IMGUI_DIR}/imgui.h
    ${SGL_IMGUI_DIR}/imgui_internal.h
    ${SGL_IMGUI_DIR}/imstb_rectpack.h
    ${SGL_IMGUI_DIR}/imstb_textedit.h
    ${SGL_IMGUI_DIR}/imstb_truetype.h
    ${SGL_IMGUI_DIR}/imgui.cpp
    ${SGL_IMGUI_DIR}/imgui_draw.cpp
    ${SGL_IMGUI_DIR}/imgui_widgets.cpp
    ${SGL_IMGUI_DIR}/imgui_tables.cpp
)

target_include_directories(sgl PUBLIC ../ ${SGL_IMGUI_DIR})

target_compile_features(sgl PUBLIC cxx_std_20)

target_compile_options(sgl
    PUBLIC
        # MSVC flags.
        $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:
            /MP                             # enable multi-processor compilation
            /Zi                             # generate debug symbols
            # Configure warnings
            /WX                             # warnings as errors
            /W4                             # increase warning level
            /wd4251                         # 'type' : class 'type1' needs to have dll-interface to be used by clients of class 'type2'
            /wd4201                         # nonstandard extension used: nameless struct/union
            /wd4189                         # local variable is initialized but not referenced
            /wd4275                         # non dll-interface class 'type' used as base for dll-interface class 'type2'
            /external:templates-            # Allows warnings from external headers when they occur in a template that's instantiated in your code.
            # Enable warnings that are disabled with /W4
            /we4263                         # 'function': member function does not override any base class virtual member function
            /we4264                         # 'virtual_function': no override available for virtual member function from base 'class'; function is hidden
            /we5038                         # data member 'member1' will be initialized after data member 'member2'
            /we5259                         # 'specialized-type': explicit specialization requires 'template <>'
            /we5263                         # calling 'std::move' on a temporary object prevents copy elision
        >
        # Clang/GCC flags.
        $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:
            -fms-extensions                 # enable MS extensions (among other things allow anonymous structs)
            -fvisibility=hidden             # hide symbols by default
            -Wall                           # set warning level
            -Wextra                         # enable extra warnings
            -Wno-unused-function
            -Wno-unused-variable
            -Wno-unused-but-set-variable
            -Wno-sign-compare
            -Wno-missing-field-initializers

            # this is a workaround for warnings in the nanobind binding code
            -Wno-odr
        >
        # Clang flags.
        $<$<CXX_COMPILER_ID:Clang>:
            -Wno-unused-private-field
            -Wno-braced-scalar-init
            -Wno-self-assign-overloaded
            -Wno-deprecated-copy
        >
        # GCC flags.
        $<$<CXX_COMPILER_ID:GNU>:
            -Wno-literal-suffix
            -Wno-class-memaccess
            -Wno-strict-aliasing
            -Wno-maybe-uninitialized
            -Wno-stringop-truncation
        >
)

if(SGL_ENABLE_COVERAGE)
    find_program(GCOV_PATH gcov)
    if(GCOV_PATH)
        target_compile_options(sgl PUBLIC $<$<CXX_COMPILER_ID:GNU>:--coverage -fprofile-abs-path>)
        target_link_options(sgl PUBLIC $<$<CXX_COMPILER_ID:GNU>:--coverage>)
        target_link_libraries(sgl PUBLIC $<$<CXX_COMPILER_ID:GNU>:gcov>)
    else()
        message(WARNING "gcov not found! Code coverage will not be available.")
    endif()
endif()

if(SGL_ENABLE_ASAN)
    target_compile_options(sgl
        PUBLIC
            $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:
                /fsanitize=address
            >
            $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:
                -fsanitize=address
            >
    )
    target_link_options(sgl
        PUBLIC
            $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:
                -fsanitize=address
            >
    )
    target_compile_definitions(sgl
        PUBLIC
            $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:
                _DISABLE_VECTOR_ANNOTATION
                _DISABLE_STRING_ANNOTATION
            >
    )
endif()

if(SGL_ENABLE_PCH)
    target_precompile_headers(sgl
        PRIVATE
            sgl/sgl_pch.h
    )
endif()

target_link_options(sgl
    PUBLIC
        # MSVC flags.
        $<$<CXX_COMPILER_ID:MSVC>:/DEBUG>           # generate debug information
)

target_compile_definitions(sgl
    PUBLIC
        SGL_DEBUG=$<BOOL:$<CONFIG:Debug>>
        # Always enable asserts unless SGL_DISABLE_ASSERTS is set.
        SGL_ENABLE_ASSERTS=$<NOT:$<BOOL:${SGL_DISABLE_ASSERTS}>>
        # Windows.
        $<$<PLATFORM_ID:Windows>:NOMINMAX>  # do not define min/max macros
        $<$<PLATFORM_ID:Windows>:UNICODE>   # force character map to unicode
        # MSVC C++ library.
        $<$<CXX_COMPILER_ID:MSVC>:_USE_MATH_DEFINES>
        $<$<CXX_COMPILER_ID:MSVC>:_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR>
        # $<$<CXX_COMPILER_ID:MSVC>:_SCL_SECURE_NO_WARNINGS>
        # $<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>
        # $<$<CXX_COMPILER_ID:MSVC>:_SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING>
        # Clang.
        $<$<CXX_COMPILER_ID:Clang>:_MSC_EXTENSIONS> # enable MS extensions
        IMGUI_USER_CONFIG="sgl/ui/imgui_config.h"
    PRIVATE
        SGL_DLL
        $<$<PLATFORM_ID:Windows>:IMGUI_API=__declspec\(dllexport\)>
        $<$<PLATFORM_ID:Linux>:IMGUI_API=__attribute__\(\(visibility\("default"\)\)\)>
        $<$<PLATFORM_ID:Darwin>:IMGUI_API=__attribute__\(\(visibility\("default"\)\)\)>
        # Linux.
        $<$<PLATFORM_ID:Linux>:__USE_LARGEFILE64>
        SGL_PROJECT_DIR="${SGL_PROJECT_DIR}"
)

file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${SGL_BUILD_TYPE}/include/sgl/core/config.h CONTENT
"#pragma once

#define SGL_BUILD_TYPE \"${SGL_BUILD_TYPE}\"

#define SGL_USE_DYNAMIC_CUDA $<BOOL:${SGL_USE_DYNAMIC_CUDA}>

#define SGL_HAS_D3D12 $<BOOL:${SGL_HAS_D3D12}>
#define SGL_HAS_VULKAN $<BOOL:${SGL_HAS_VULKAN}>

#define SGL_HAS_NVAPI $<BOOL:${SGL_HAS_NVAPI}>
#define SGL_HAS_AGILITY_SDK $<BOOL:${SGL_HAS_AGILITY_SDK}>

#define SGL_HAS_LIBJPEG $<BOOL:${SGL_HAS_LIBJPEG}>
#define SGL_HAS_LIBPNG $<BOOL:${SGL_HAS_LIBPNG}>
#define SGL_HAS_OPENEXR $<BOOL:${SGL_HAS_OPENEXR}>
#define SGL_HAS_ASMJIT $<BOOL:${SGL_HAS_ASMJIT}>
"
)
target_include_directories(sgl PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/${SGL_BUILD_TYPE}/include)

target_link_libraries(sgl
    PUBLIC
        fmt::fmt
        slang
        slang-rhi
        slang-rhi-vulkan-headers
        header_only
        $<$<NOT:$<BOOL:${SGL_USE_DYNAMIC_CUDA}>>:CUDA::cuda_driver>
        sgl_data
    PRIVATE
        git_version
        glfw
        tevclient
        $<$<BOOL:${SGL_HAS_D3D12}>:d3d12>
        $<$<BOOL:${SGL_HAS_LIBPNG}>:PNG::PNG>
        $<$<BOOL:${SGL_HAS_LIBJPEG}>:JPEG::JPEG>
        $<$<BOOL:${SGL_HAS_OPENEXR}>:OpenEXR::OpenEXR>
        $<$<BOOL:${SGL_HAS_ASMJIT}>:asmjit::asmjit>
        # Windows system libraries.
        $<$<PLATFORM_ID:Windows>:Dbghelp>
        # $<$<PLATFORM_ID:Windows>:shcore.lib>
        # $<$<PLATFORM_ID:Windows>:shlwapi.lib>
        # $<$<PLATFORM_ID:Windows>:comctl32.lib>
        # $<$<PLATFORM_ID:Windows>:setupapi.lib>  # Used in MonitorInfo
        # Linux system libraries.
        # $<$<PLATFORM_ID:Linux>:gtk3>
)

set_target_properties(sgl PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${SGL_RUNTIME_OUTPUT_DIRECTORY}
    LIBRARY_OUTPUT_DIRECTORY ${SGL_LIBRARY_OUTPUT_DIRECTORY}
)

sgl_header_validation(sgl)

target_copy_shaders(sgl sgl)

get_property(shader_files_src GLOBAL PROPERTY SGL_SHADER_FILES_SRC)
get_property(shader_files_dst GLOBAL PROPERTY SGL_SHADER_FILES_DST)

foreach(file IN ZIP_LISTS shader_files_src shader_files_dst)
    # if dst is directory create it
    add_custom_command(TARGET sgl POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ${file_1} COMMAND ${CMAKE_COMMAND} -E copy_if_different ${file_0} ${file_1}/)
endforeach()
