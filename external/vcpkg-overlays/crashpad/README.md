This overlay of crashpad is based on the crashpad port found in vcpkg 2025.08.27.

The only thing it adds is a new patch applied to `mini_chromium` to fix finding VS build tools.

Changes:
- Add patch in `fix-find-vs-build-tools.patch`
- Modify `portfile.cmake` to add `fix-find-vs-build-tools.patch` to the `PATCHES` section for the `vcpkg_from_git` checking out mini_chromium
