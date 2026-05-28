set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)
# _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR avoids the use of constexpr mutex constructor
# in vcpkg packages, which can lead to binary incompatibility issues.
set(VCPKG_C_FLAGS "-D_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR")
set(VCPKG_CXX_FLAGS "-D_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR")
