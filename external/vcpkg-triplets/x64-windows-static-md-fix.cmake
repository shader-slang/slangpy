# Allow user to explicitly specify a toolset version via -DSGL_MSVC_TOOLSET_VERSION=...
# If not specified, it falls back to the VCToolsVersion from the developer environment.
if(DEFINED SGL_MSVC_TOOLSET_VERSION)
    set(toolset_to_use ${SGL_MSVC_TOOLSET_VERSION})
    message(STATUS "Triplet: Using user-specified toolset version '${toolset_to_use}'.")
elseif(DEFINED ENV{VCToolsVersion})
    set(toolset_to_use "$ENV{VCToolsVersion}")
    message(STATUS "Triplet: Using toolset version from environment: '${toolset_to_use}'.")
endif()

if(DEFINED toolset_to_use)
    # Set the detailed toolset version. This forces vcpkg to use the specific toolset.
    set(VCPKG_PLATFORM_TOOLSET_VERSION "${toolset_to_use}")

    # Also set the major toolset version, as vcpkg may require it to be present.
    string(REGEX MATCH "^([0-9]+)\\.([0-9])" _match "${toolset_to_use}")
    if(_match)
        set(derived_toolset "v${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
        set(VCPKG_PLATFORM_TOOLSET "${derived_toolset}")
    endif()
endif()

set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)
# _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR avoids the use of constexpr mutex constructor
# in vcpkg packages, which can lead to binary incompatibility issues.
set(VCPKG_C_FLAGS "-D_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR")
set(VCPKG_CXX_FLAGS "-D_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR")
