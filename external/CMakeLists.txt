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

# slang

set(SGL_SLANG_DEBUG_INFO ON CACHE BOOL "Download debug info for slang")

set(SGL_LOCAL_SLANG OFF CACHE BOOL "Use a local build of slang instead of downloading a prebuilt binary")
set(SGL_LOCAL_SLANG_DIR "${CMAKE_SOURCE_DIR}/../slang" CACHE PATH "Path to a local slang build")
set(SGL_LOCAL_SLANG_BUILD_DIR "build/Debug" CACHE STRING "Build directory of the local slang build")

set(SLANG_VERSION "2025.17")
set(SLANG_URL_BASE "https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/slang-${SLANG_VERSION}")

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

sgl_download_package(slang ${SLANG_URL})

set(SLANG_DIR ${slang_SOURCE_DIR})
set(SLANG_INCLUDE_DIR ${slang_SOURCE_DIR}/include)

if(SGL_LOCAL_SLANG)
    set(SLANG_DIR ${SGL_LOCAL_SLANG_DIR}/${SGL_LOCAL_SLANG_BUILD_DIR})
    set(SLANG_INCLUDE_DIR ${SGL_LOCAL_SLANG_DIR}/include)
endif()

if(SGL_WINDOWS)
    add_library(slang SHARED IMPORTED GLOBAL)
    set_target_properties(slang PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${SLANG_INCLUDE_DIR}
        IMPORTED_IMPLIB ${SLANG_DIR}/lib/slang.lib
        IMPORTED_LOCATION ${SLANG_DIR}/bin/slang.dll
    )
    sgl_copy_binary(${SLANG_DIR}/bin/slang.dll .)
    sgl_copy_binary(${SLANG_DIR}/bin/slang-glslang.dll .)
    sgl_copy_binary(${SLANG_DIR}/bin/slang-glsl-module.dll .)
    sgl_copy_binary(${SLANG_DIR}/bin/slang-llvm.dll .)
    sgl_copy_binary(${SLANG_DIR}/bin/slang-rt.dll .)
elseif(SGL_LINUX)
    add_library(slang SHARED IMPORTED GLOBAL)
    set_target_properties(slang PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${SLANG_INCLUDE_DIR}
        IMPORTED_LOCATION ${SLANG_DIR}/lib/libslang.so
    )
    sgl_copy_binary(${SLANG_DIR}/lib/libslang.so .)
    sgl_copy_binary(${SLANG_DIR}/lib/libslang-glslang.so .)
    sgl_copy_binary(${SLANG_DIR}/lib/libslang-glsl-module.so .)
    # sgl_copy_binary(${SLANG_DIR}/lib/libslang-llvm.so .)
    sgl_copy_binary(${SLANG_DIR}/lib/libslang-rt.so .)
elseif(SGL_MACOS)
    add_library(slang SHARED IMPORTED GLOBAL)
    set_target_properties(slang PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${SLANG_INCLUDE_DIR}
        IMPORTED_LOCATION ${SLANG_DIR}/lib/libslang.dylib
    )
    sgl_copy_binary(${SLANG_DIR}/lib/libslang.dylib .)
    sgl_copy_binary(${SLANG_DIR}/lib/libslang-glslang.dylib .)
    sgl_copy_binary(${SLANG_DIR}/lib/libslang-glsl-module.dylib .)
    # sgl_copy_binary(${SLANG_DIR}/lib/libslang-llvm.dylib .)
    sgl_copy_binary(${SLANG_DIR}/lib/libslang-rt.dylib .)
endif()

if(NOT SGL_LOCAL_SLANG AND SGL_SLANG_DEBUG_INFO)
    sgl_download_package(slang_debug_info ${SLANG_DEBUG_INFO_URL})
    if(SGL_WINDOWS)
        # Note: The downloaded archive has single top-level 'bin' folder that is stripped out by FetchContent.
        sgl_copy_binary(${slang_debug_info_SOURCE_DIR}/slang.pdb .)
        sgl_copy_binary(${slang_debug_info_SOURCE_DIR}/slang-rt.pdb .)
    elseif(SGL_LINUX)
        sgl_copy_binary(${slang_debug_info_SOURCE_DIR}/lib/libslang.so.dwarf .)
        sgl_copy_binary(${slang_debug_info_SOURCE_DIR}/lib/libslang-rt.so.dwarf .)
    elseif(SGL_MACOS)
        sgl_copy_binary(${slang_debug_info_SOURCE_DIR}/lib/libslang.dylib.dwarf .)
        sgl_copy_binary(${slang_debug_info_SOURCE_DIR}/lib/libslang-rt.dylib.dwarf .)
    endif()
endif()

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

if(TARGET slang-rhi-optix)
    get_target_property(OPTIX_DIR slang-rhi-optix INTERFACE_INCLUDE_DIRECTORIES)
    sgl_copy_binary(${OPTIX_DIR}/optix.h optix)
    sgl_copy_binary(${OPTIX_DIR}/optix_device.h optix)
    sgl_copy_binary(${OPTIX_DIR}/optix_types.h optix)
    sgl_copy_binary(${OPTIX_DIR}/internal/optix_device_impl.h optix/internal)
    sgl_copy_binary(${OPTIX_DIR}/internal/optix_device_impl_transformations.h optix/internal)
    set(SGL_HAS_OPTIX ON PARENT_SCOPE)
else()
    set(SGL_HAS_OPTIX OFF PARENT_SCOPE)
endif()

# imgui

set(IMGUI_VERSION "1.90.1")
sgl_download_package(imgui "https://github.com/ocornut/imgui/archive/refs/tags/v${IMGUI_VERSION}.zip")
set(SGL_IMGUI_DIR ${imgui_SOURCE_DIR} PARENT_SCOPE)

# header only

add_library(header_only INTERFACE)
target_include_directories(header_only INTERFACE include)
