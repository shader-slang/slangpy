# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)

# Use FetchContent to download and populate a package without calling add_subdirectory().
# This is used for downloading prebuilt external binaries.
macro(sgl_download_package name url)
    FetchContent_Declare(
        ${name}
        URL ${url}
        SOURCE_SUBDIR _does_not_exist_ # avoid adding contained CMakeLists.txt
        HTTP_HEADER "Authorization: token ${SLANG_GITHUB_TOKEN}"
    )
    FetchContent_GetProperties(${name})
    if(NOT ${name}_POPULATED)
        message(STATUS "Populating ${name} ...")
        FetchContent_MakeAvailable(${name})
    endif()
endmacro()

# d3d12

add_library(d3d12 INTERFACE)
target_link_libraries(d3d12 INTERFACE dxgi.lib d3d12.lib dxguid.lib)

# fmt

set(FMT_INSTALL OFF)
add_subdirectory(fmt)

# glfw

set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
set(GLFW_BUILD_DOCS OFF)
set(GLFW_INSTALL OFF)
add_subdirectory(glfw)

# tevclient

add_subdirectory(tevclient EXCLUDE_FROM_ALL)

# nanobind

if(SGL_BUILD_PYTHON)
    add_subdirectory(nanobind)
    nanobind_build_library(nanobind SHARED)
endif()

# nanothread

set(NANOTHREAD_STATIC ON)
add_subdirectory(nanothread)
# Disable LTO for nanothread as this causes problems when linking with sgl.
set_target_properties(nanothread PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE FALSE)

# lmdb

add_library(lmdb STATIC)
target_include_directories(lmdb INTERFACE lmdb)
target_sources(lmdb PRIVATE lmdb/mdb.c lmdb/midl.c)
target_compile_definitions(lmdb PRIVATE _CRT_SECURE_NO_WARNINGS)
target_compile_options(lmdb PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/wd4333 /wd4172>)
target_compile_options(lmdb PRIVATE $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-Wno-deprecated-declarations>)

# slang

set(SGL_SLANG_DEBUG_INFO ON CACHE BOOL "Download debug info for slang")

set(SGL_LOCAL_SLANG OFF CACHE BOOL "Use a local build of slang instead of downloading a prebuilt binary")
set(SGL_LOCAL_SLANG_DIR "${CMAKE_SOURCE_DIR}/../slang" CACHE PATH "Path to a local slang build")
set(SGL_LOCAL_SLANG_BUILD_DIR "build/Debug" CACHE STRING "Build directory of the local slang build")

set(SLANG_VERSION "2025.24.3")
set(SLANG_URL_BASE "https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/slang-${SLANG_VERSION}")

set(SLANG_URL_BASE "https://github.com/shader-slang/slang/releases/download/v${SGL_SLANG_VERSION}/slang-${SGL_SLANG_VERSION}")

if(SGL_WINDOWS)
    if(SGL_ARCHITECTURE MATCHES "x86_64")
        set(SLANG_URL "${SLANG_URL_BASE}-windows-x86_64.zip")
        set(SLANG_DEBUG_INFO_URL "${SLANG_URL_BASE}-windows-x86_64-debug-info.zip")
    elseif(SGL_ARCHITECTURE MATCHES "aarch64|arm64")
        set(SLANG_URL "${SLANG_URL_BASE}-windows-aarch64.zip")
        set(SLANG_DEBUG_INFO_URL "${SLANG_URL_BASE}-windows-aarch64-debug-info.zip")
    endif()
elseif(SGL_LINUX)
    if(SGL_ARCHITECTURE MATCHES "x86_64")
        set(SLANG_URL "${SLANG_URL_BASE}-linux-x86_64.tar.gz")
        set(SLANG_DEBUG_INFO_URL "${SLANG_URL_BASE}-linux-x86_64-debug-info.tar.gz")
    elseif(SGL_ARCHITECTURE MATCHES "aarch64|arm64")
        set(SLANG_URL "${SLANG_URL_BASE}-linux-aarch64.tar.gz")
        set(SLANG_DEBUG_INFO_URL "${SLANG_URL_BASE}-linux-aarch64-debug-info.tar.gz")
    endif()
elseif(SGL_MACOS)
    if(CMAKE_APPLE_SILICON_PROCESSOR MATCHES "x86_64")
        set(SLANG_URL "${SLANG_URL_BASE}-macos-x86_64.zip")
        set(SLANG_DEBUG_INFO_URL "${SLANG_URL_BASE}-macos-x86_64-debug-info.zip")
    else()
        set(SLANG_URL "${SLANG_URL_BASE}-macos-aarch64.zip")
        set(SLANG_DEBUG_INFO_URL "${SLANG_URL_BASE}-macos-aarch64-debug-info.zip")
    endif()
endif()

if(NOT SGL_LOCAL_SLANG)
    sgl_download_package(slang ${SLANG_URL})
endif()

set(SLANG_DIR ${slang_SOURCE_DIR})
set(SLANG_INCLUDE_DIR ${slang_SOURCE_DIR}/include)

if(SGL_LOCAL_SLANG)
    set(SLANG_DIR ${SGL_LOCAL_SLANG_DIR}/${SGL_LOCAL_SLANG_BUILD_DIR})
    set(SLANG_INCLUDE_DIR ${SGL_LOCAL_SLANG_DIR}/include)
endif()

# Slang library adding logic to find the exact filenames to use and copy
function(sgl_add_library_slang)
    if(SGL_WINDOWS)
        set(slang_lib_dir "${SLANG_DIR}/bin")
        set(slang_lib_prefix "")
        set(slang_lib_suffix "dll")
    else()
        set(slang_lib_dir "${SLANG_DIR}/lib")
        set(slang_lib_prefix "lib")
        if(SGL_MACOS)
            set(slang_lib_suffix "dylib")
        else()
            set(slang_lib_suffix "so")
        endif()
    endif()

    # Determine whether to use versioned slang-compiler or old name
    if(EXISTS ${slang_lib_dir}/${slang_lib_prefix}slang-compiler.${slang_lib_suffix})
        set(slang_lib_base slang-compiler)
        if(SGL_LOCAL_SLANG)
            # Extract slang version from local build's tag-version header
            # Only applicable for versioned slang builds (since v2025.21)
            set(SLANG_TAG_VERSION_HEADER "${SLANG_DIR}/include/slang-tag-version.h")
            if(EXISTS "${SLANG_TAG_VERSION_HEADER}")
                extract_define_from_header("${SLANG_TAG_VERSION_HEADER}" "SLANG_VERSION_NUMERIC" SLANG_VERSION_FROM_HEADER)
                if(SLANG_VERSION_FROM_HEADER)
                    set(SGL_SLANG_VERSION "${SLANG_VERSION_FROM_HEADER}")
                    message(STATUS "Using Slang version from local build: ${SGL_SLANG_VERSION}")
                else()
                    message(FATAL_ERROR "Could not extract SLANG_VERSION_NUMERIC from: ${SLANG_TAG_VERSION_HEADER}")
                endif()
            else()
                message(FATAL_ERROR "Slang tag version header not found: ${SLANG_TAG_VERSION_HEADER}")
            endif()
        endif()
        set(slang_module_version "-${SGL_SLANG_VERSION}")
    else()
        set(slang_lib_base slang)
        set(slang_module_version "")
    endif()

    add_library(slang SHARED IMPORTED GLOBAL)
    if(SGL_WINDOWS)
        # Windows filenames are not versioned
        set_target_properties(slang PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES ${SLANG_INCLUDE_DIR}
            IMPORTED_IMPLIB ${SLANG_DIR}/lib/${slang_lib_base}.lib
            IMPORTED_LOCATION ${SLANG_DIR}/bin/${slang_lib_base}.dll
        )
        sgl_copy_binary(${SLANG_DIR}/bin/${slang_lib_base}.dll .)
        sgl_copy_binary(${SLANG_DIR}/bin/slang-glslang.dll .)
        sgl_copy_binary(${SLANG_DIR}/bin/slang-glsl-module.dll .)
        # slang-llvm.dll is available in x64 prebuilt packages but not in ARM64 packages.
        # Use conditional copy to support both architectures.
        if(EXISTS ${SLANG_DIR}/bin/slang-llvm.dll)
            sgl_copy_binary(${SLANG_DIR}/bin/slang-llvm.dll .)
        endif()
        # Note: The downloaded debug-info archive has single top-level 'bin'
        # folder that is stripped out by FetchContent.
        set(slang_resolved_debug_info "${slang_lib_base}.pdb")
    else()
        # Python wheels don't support symlinks, so we need to find the actual
        # library files.
        file(REAL_PATH "${slang_lib_dir}/${slang_lib_prefix}${slang_lib_base}.${slang_lib_suffix}" slang_resolved)
        set_target_properties(slang PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES ${SLANG_INCLUDE_DIR}
            IMPORTED_LOCATION ${slang_resolved}
        )
        sgl_copy_binary(${slang_resolved} .)
        sgl_copy_binary(${slang_lib_dir}/libslang-glslang${slang_module_version}.${slang_lib_suffix} .)
        sgl_copy_binary(${slang_lib_dir}/libslang-glsl-module${slang_module_version}.${slang_lib_suffix} .)
        # sgl_copy_binary(${slang_lib_dir}/libslang-llvm.${slang_lib_suffix} .)
        get_filename_component(slang_resolved_debug_info_base ${slang_resolved} NAME)
        set(slang_resolved_debug_info "lib/${slang_resolved_debug_info_base}.dwarf")
    endif()

    if(NOT SGL_LOCAL_SLANG AND SGL_SLANG_DEBUG_INFO)
        sgl_download_package(slang_debug_info ${SLANG_DEBUG_INFO_URL})
        sgl_copy_binary(${slang_debug_info_SOURCE_DIR}/${slang_resolved_debug_info} .)
    endif()
endfunction()

sgl_add_library_slang()

# Install slang header files
install(FILES ${SLANG_DIR}/include/slang.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${SLANG_DIR}/include/slang-deprecated.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${SLANG_DIR}/include/slang-image-format-defs.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# slang-rhi

set(SLANG_RHI_BUILD_TESTS OFF)
set(SLANG_RHI_BUILD_EXAMPLES OFF)
set(SLANG_RHI_FETCH_SLANG OFF)
set(SLANG_RHI_ENABLE_D3D11 OFF)
set(SLANG_RHI_ENABLE_WGPU OFF)
# TODO: We should rename SLANG_RHI_BUILD_FROM_SLANG_REPO to something more generic.
# We want slang-rhi to support using a `slang` target that is provided from the outside.
set(SLANG_RHI_BUILD_FROM_SLANG_REPO ON)
set(SLANG_RHI_FETCH_SLANG OFF)
set(SLANG_RHI_SLANG_INCLUDE_DIR ${SLANG_INCLUDE_DIR} CACHE STRING "" FORCE)
set(SLANG_RHI_SLANG_BINARY_DIR ${SLANG_DIR} CACHE STRING "" FORCE)
add_subdirectory(slang-rhi)

# Copy shared slang-rhi-device.h header to shaders directory (wrapped in sgl/device/rhi.slang)
sgl_copy_shader_file(${CMAKE_CURRENT_SOURCE_DIR}/slang-rhi/include/slang-rhi-device.h slang-rhi)

if(TARGET slang-rhi-nvapi)
    get_target_property(NVAPI_DIR slang-rhi-nvapi INTERFACE_INCLUDE_DIRECTORIES)
    sgl_copy_shader_file(${NVAPI_DIR}/nvHLSLExtns.h nvapi)
    sgl_copy_shader_file(${NVAPI_DIR}/nvHLSLExtnsInternal.h nvapi)
    sgl_copy_shader_file(${NVAPI_DIR}/nvShaderExtnEnums.h nvapi)
    set(SGL_HAS_NVAPI ON PARENT_SCOPE)
else()
    set(SGL_HAS_NVAPI OFF PARENT_SCOPE)
endif()

if(TARGET slang-rhi-agility-sdk)
    set(SGL_HAS_AGILITY_SDK ON PARENT_SCOPE)
else()
    set(SGL_HAS_AGILITY_SDK OFF PARENT_SCOPE)
endif()

# imgui

set(IMGUI_VERSION "1.90.1")
sgl_download_package(imgui "https://github.com/ocornut/imgui/archive/refs/tags/v${IMGUI_VERSION}.zip")
set(SGL_IMGUI_DIR ${imgui_SOURCE_DIR} PARENT_SCOPE)

# header only

add_library(header_only INTERFACE)
target_include_directories(header_only INTERFACE include)
