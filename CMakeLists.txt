cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0077 NEW)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(utils)

set(VCPKG_INSTALL_OPTIONS "--no-print-usage" "--allow-unsupported")

project(sgl)

set(CTEST_OUTPUT_ON_FAILURE TRUE)
enable_testing()

include(DetermineTargetArchitecture)
include(GNUInstallDirs)
include(CMakeRC)

set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0")

if (APPLE)
  set(SGL_ORIGIN "@loader_path")
elseif(UNIX)
  set(SGL_ORIGIN "$ORIGIN")
endif()

# TODO we should probably set install RPATH on individual targets instead
set(CMAKE_INSTALL_RPATH "${SGL_ORIGIN};${SGL_ORIGIN}/../lib")

# Check if this project is the master cmake project (i.e. not included via add_subdirectory).
set(SGL_MASTER_PROJECT OFF)
if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
  set(SGL_MASTER_PROJECT ON)
  set(SGL_BUILD_STANDALONE OFF)
endif()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

option(SGL_BUILD_PYTHON "Build Python extension" ON)
option(SGL_BUILD_EXAMPLES "Build example applications" ON)
option(SGL_BUILD_TESTS "Build tests" ON)
option(SGL_BUILD_DOC "Build documentation" OFF)
option(SGL_GENERATE_SETPATH_SCRIPTS "Generate setpath scripts" ON)

# Enable building sgl without external dependencies.
# This is useful when using sgl through cmake add_subdirectory.
option(SGL_BUILD_STANDALONE "Build without external dependencies" ON)

# Enable linking CUDA driver API dynamically.
# This allows to build CUDA support without having CUDA SDK installed.
option(SGL_USE_DYNAMIC_CUDA "Load CUDA driver API dynamically" ON)

# Disable asserts.
# By default, asserts are enabled in both debug and release builds.
# This option can be used to override the default behavior.
option(SGL_DISABLE_ASSERTS "Disable asserts" OFF)

# Enable/disable precompiled headers.
option(SGL_ENABLE_PCH "Enable precompiled headers" OFF)

# Enable/disable code coverage.
option(SGL_ENABLE_COVERAGE "Enable code coverage (gcov)" OFF)

# Enable/disable address sanitizer.
option(SGL_ENABLE_ASAN "Enable address sanitizer" OFF)

# Enable/disable header validation.
# If enabled, additional targets are generated to validate that headers are self sufficient.
option(SGL_ENABLE_HEADER_VALIDATION "Enable header validation" OFF)

set(SGL_PROJECT_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE STRING "Project directory (defaults to CMAKE_CURRENT_SOURCE_DIR)")

set(SGL_BINARY_DIR ${CMAKE_BINARY_DIR} CACHE STRING "Binary directory (defaults to CMAKE_BINARY_DIR)")

# -----------------------------------------------------------------------------
# Check platform
# -----------------------------------------------------------------------------

if(${CMAKE_SYSTEM_NAME} MATCHES "Window")
    set(SGL_PLATFORM "Windows")
    set(SGL_WINDOWS TRUE)
    set(SGL_HAS_D3D12 ON)
    set(SGL_HAS_VULKAN ON)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(SGL_PLATFORM "Linux")
    set(SGL_LINUX TRUE)
    set(SGL_HAS_D3D12 OFF)
    set(SGL_HAS_VULKAN ON)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(SGL_PLATFORM "macOS")
    set(SGL_MACOS TRUE)
    set(SGL_HAS_D3D12 OFF)
    set(SGL_HAS_VULKAN ON)
else()
    message(FATAL_ERROR "Unsupported platform!")
endif()
message(STATUS "Platform: ${SGL_PLATFORM}")

# Determine the target architecture we build for.
# CMAKE_SYSTEM_PROCESSOR is not a reliable way to determine the target architecture.
determine_target_architecture(SGL_ARCHITECTURE)
message(STATUS "Architecture: ${SGL_ARCHITECTURE}")

# -----------------------------------------------------------------------------
# Python
# -----------------------------------------------------------------------------

if(SGL_BUILD_PYTHON)
    find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
endif()

# -----------------------------------------------------------------------------
# CUDA
# -----------------------------------------------------------------------------

if(NOT SGL_USE_DYNAMIC_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# -----------------------------------------------------------------------------
# Global setup
# -----------------------------------------------------------------------------

# Require builds to be outside of source tree.
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
    message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please use a build directory instead.")
endif()

# Enable folders (for Visual Studio).
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Setup available build configurations.
if(NOT SETUP_CONFIGURATION_TYPES)
    set(SETUP_CONFIGURATION_TYPES 1)

    if(CMAKE_CONFIGURATION_TYPES)
        # multi config generator
        set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING "" FORCE)
    else()
        # single config generator
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY HELPSTRING "Choose the type of build")
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug;Release;RelWithDebInfo")
    endif()
endif()

set(SGL_RUNTIME_OUTPUT_DIRECTORY ${SGL_BINARY_DIR})
set(SGL_LIBRARY_OUTPUT_DIRECTORY ${SGL_BINARY_DIR})

if(CMAKE_CONFIGURATION_TYPES)
    set(SGL_BUILD_TYPE $<CONFIG>)
    set(SGL_OUTPUT_DIRECTORY ${SGL_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>)
else()
    set(SGL_BUILD_TYPE ${CMAKE_BUILD_TYPE})
    set(SGL_OUTPUT_DIRECTORY ${SGL_RUNTIME_OUTPUT_DIRECTORY})
endif()

set(SGL_SHADER_OUTPUT_DIRECTORY ${SGL_OUTPUT_DIRECTORY}/shaders)
# set(SGL_PLUGIN_OUTPUT_DIRECTORY ${SGL_OUTPUT_DIRECTORY}/plugins)

# -----------------------------------------------------------------------------
# Version
# -----------------------------------------------------------------------------

# Embed git information.
include(git_version)
git_version_setup()

# -----------------------------------------------------------------------------
# Macros
# -----------------------------------------------------------------------------

macro(sgl_add_example target)
    add_executable(${target})
    target_link_libraries(${target} PRIVATE sgl)
    target_compile_definitions(${target} PRIVATE SGL_EXAMPLE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${SGL_RUNTIME_OUTPUT_DIRECTORY})
    set_target_properties(${target} PROPERTIES FOLDER "examples")
endmacro()

# -----------------------------------------------------------------------------
# Header validation
# -----------------------------------------------------------------------------

# Helper function to validate header files of a target to be self sufficient.
# This creates a new target with "_hv" suffix that compiles one
# translation unit for each header file found in the supplied targets sources list.
function(sgl_header_validation target)
    if(NOT SGL_ENABLE_HEADER_VALIDATION)
        return()
    endif()

    cmake_parse_arguments(PARSE_ARGV 1 args "" "" "IGNORE")
    set(ignore_files ${args_IGNORE})

    message(STATUS "Setting up header validation for target: ${target}")

    # Create a new target for validating headers.
    set(validate_target ${target}_hv)
    add_library(${validate_target} OBJECT)

    # Copy properties from original target.
    foreach(property INCLUDE_DIRECTORIES LINK_LIBRARIES COMPILE_DEFINITIONS COMPILE_FEATURES)
        get_target_property(value ${target} ${property})
        if(value)
            set_property(TARGET ${validate_target} PROPERTY ${property} ${value})
        endif()
    endforeach()

    # Add target source directory as an include directory.
    get_target_property(target_source_dir ${target} SOURCE_DIR)
    target_include_directories(${validate_target} PRIVATE ${target_source_dir})

    # Get list of source files.
    get_target_property(sources ${target} SOURCES)

    # Create a list of CPP files, each including one header.
    set(validate_sources "")
    foreach(file ${sources})
        if(${file} MATCHES "^[^.][^.].*\\.h$" AND NOT ${file} IN_LIST ignore_files)
            string(SHA1 id ${file})
            # shorten to help avoid super long file names
            string(SUBSTRING ${id} 0, 8, id)
            set(cpp_file ${validate_target}/${id}.cpp)
            set(cpp_content "#include \"${file}\"\n")
            file(CONFIGURE OUTPUT ${cpp_file} CONTENT ${cpp_content})
            list(APPEND validate_sources ${CMAKE_CURRENT_BINARY_DIR}/${cpp_file})
        endif()
    endforeach()

    target_compile_options(${validate_target} PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/bigobj>)
    target_compile_definitions(${validate_target} PRIVATE SGL_HEADER_VALIDATION)

    target_sources(${validate_target} PRIVATE ${validate_sources})
    set_target_properties(${validate_target} PROPERTIES FOLDER "Validation")
endfunction()

# -----------------------------------------------------------------------------
# Shader files
# -----------------------------------------------------------------------------

set_property(GLOBAL PROPERTY SGL_SHADER_FILES_SRC "")
set_property(GLOBAL PROPERTY SGL_SHADER_FILES_DST "")

function(sgl_copy_shader_file src dst_dir)
    message(STATUS "Copying shader file ${src} to ${dst_dir}")
    set(dst ${SGL_SHADER_OUTPUT_DIRECTORY}/${dst_dir})
    set_property(GLOBAL APPEND PROPERTY SGL_SHADER_FILES_SRC ${src})
    set_property(GLOBAL APPEND PROPERTY SGL_SHADER_FILES_DST ${dst})
    install(FILES ${src} DESTINATION ${CMAKE_INSTALL_DATADIR}/shaders/${dst_dir})
endfunction()

# Regex for matching shader files.
set(SHADER_EXTENSION_REGEX "\.(slang|slangh|hlsli|hlsl)$")

# Setup build rules to copy all shaders of a target to the output directory.
# The specified output_dir is relative to the global shader output directory (SGL_SHADER_OUTPUT_DIRECTORY).
function(target_copy_shaders target output_dir)
    get_target_property(target_source_dir ${target} SOURCE_DIR)
    set(shader_output_dir ${SGL_SHADER_OUTPUT_DIRECTORY}/${output_dir})
    set(shader_install_dir ${CMAKE_INSTALL_DATADIR}/shaders/${output_dir})

    # Get list of all target sources.
    get_target_property(target_sources_ ${target} SOURCES)

    # Create custom commands for copying shader sources.
    foreach(file ${target_sources_})
        if(${file} MATCHES ${SHADER_EXTENSION_REGEX})
            if(IS_ABSOLUTE ${file})
                file(RELATIVE_PATH file ${target_source_dir} ${file})
            endif()

            set(src_file ${target_source_dir}/${file})
            set(dst_file ${shader_output_dir}/${file})

            add_custom_command(
                OUTPUT ${dst_file}
                COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different ${src_file} ${dst_file}
                MAIN_DEPENDENCY ${src_file}
                COMMENT "${target}: Copying shader ${file}"
            )

            get_filename_component(dir ${file} DIRECTORY)
            install(FILES ${src_file} DESTINATION ${shader_install_dir}/${dir})
        endif()
    endforeach()
endfunction()

# -----------------------------------------------------------------------------
# Binary files
# -----------------------------------------------------------------------------

set_property(GLOBAL PROPERTY SGL_BINARY_FILES_SRC "")
set_property(GLOBAL PROPERTY SGL_BINARY_FILES_DST "")

function(sgl_copy_binary src dst_dir)
    get_filename_component(filename ${src} NAME)
    set(dst ${SGL_OUTPUT_DIRECTORY}/${dst_dir}/${filename})
    set_property(GLOBAL APPEND PROPERTY SGL_BINARY_FILES_SRC ${src})
    set_property(GLOBAL APPEND PROPERTY SGL_BINARY_FILES_DST ${dst})
    install(FILES ${src} DESTINATION ${CMAKE_INSTALL_BINDIR}/${dst_dir})
endfunction()

function(sgl_copy_library target)
    get_property(src TARGET ${target} PROPERTY IMPORTED_LOCATION)
    sgl_copy_binary(${src} .)
endfunction()

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------

if(SGL_BUILD_STANDALONE)
    set(SGL_HAS_LIBJPEG OFF)
    set(SGL_HAS_LIBPNG OFF)
    set(SGL_HAS_OPENEXR OFF)
    set(SGL_HAS_ASMJIT OFF)
else()
    find_package(JPEG)
    ternary(SGL_HAS_LIBJPEG ${JPEG_FOUND} ON OFF)
    find_package(PNG)
    ternary(SGL_HAS_LIBPNG ${PNG_FOUND} ON OFF)
    find_package(OpenEXR CONFIG)
    ternary(SGL_HAS_OPENEXR ${OpenEXR_FOUND} ON OFF)
    find_package(asmjit)
    ternary(SGL_HAS_ASMJIT ${asmjit_FOUND} ON OFF)
endif()

# -----------------------------------------------------------------------------
# Embedded resources
# -----------------------------------------------------------------------------

cmrc_add_resource_library(sgl_data
    ${CMAKE_CURRENT_SOURCE_DIR}/data/fonts/Montserrat-Regular.ttf
    ${CMAKE_CURRENT_SOURCE_DIR}/data/fonts/Inconsolata-Regular.ttf
)

# -----------------------------------------------------------------------------
# Sources
# -----------------------------------------------------------------------------

add_subdirectory(external)

message(STATUS "Feature flags:")
message(STATUS "SGL_HAS_D3D12: ${SGL_HAS_D3D12}")
message(STATUS "SGL_HAS_VULKAN: ${SGL_HAS_VULKAN}")
message(STATUS "SGL_HAS_NVAPI: ${SGL_HAS_NVAPI}")
message(STATUS "SGL_HAS_AGILITY_SDK: ${SGL_HAS_AGILITY_SDK}")
message(STATUS "SGL_HAS_LIBJPEG: ${SGL_HAS_LIBJPEG}")
message(STATUS "SGL_HAS_LIBPNG: ${SGL_HAS_LIBPNG}")
message(STATUS "SGL_HAS_OPENEXR: ${SGL_HAS_OPENEXR}")
message(STATUS "SGL_HAS_ASMJIT: ${SGL_HAS_ASMJIT}")

add_subdirectory(src)

if (SGL_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(SGL_BUILD_DOC)
    add_subdirectory(docs)
endif()

if(SGL_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# -----------------------------------------------------------------------------
# copy binary files
# -----------------------------------------------------------------------------

get_property(binary_files_src GLOBAL PROPERTY SGL_BINARY_FILES_SRC)
get_property(binary_files_dst GLOBAL PROPERTY SGL_BINARY_FILES_DST)
foreach(file IN ZIP_LISTS binary_files_src binary_files_dst)
    get_filename_component(filename ${file_0} NAME)
    add_custom_command(
        OUTPUT ${file_1}
        DEPENDS ${file_0}
        COMMAND ${CMAKE_COMMAND} -E copy ${file_0} ${file_1}
        COMMENT "Copying binary file ${filename}"
    )
endforeach()

add_custom_target(copy_binary_files ALL DEPENDS ${binary_files_dst})
add_dependencies(sgl copy_binary_files)

# -----------------------------------------------------------------------------
# misc
# -----------------------------------------------------------------------------

# Generate setpath scripts.
if(SGL_GENERATE_SETPATH_SCRIPTS)
    if(SGL_WINDOWS)
        file(GENERATE OUTPUT ${SGL_OUTPUT_DIRECTORY}/setpath.bat INPUT ${CMAKE_CURRENT_SOURCE_DIR}/resources/setpath.bat.in)
        file(GENERATE OUTPUT ${SGL_OUTPUT_DIRECTORY}/setpath.ps1 INPUT ${CMAKE_CURRENT_SOURCE_DIR}/resources/setpath.ps1.in)
    endif()

    if(SGL_LINUX OR SGL_MACOS)
        file(GENERATE OUTPUT ${SGL_OUTPUT_DIRECTORY}/setpath.sh INPUT ${CMAKE_CURRENT_SOURCE_DIR}/resources/setpath.sh.in)
    endif()
endif()

# -----------------------------------------------------------------------------
# install
# -----------------------------------------------------------------------------

# Install binaries.
install(
    TARGETS sgl
    LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install header files.
install(
    DIRECTORY src/sgl
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)
