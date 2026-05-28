// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <vector>
#include <filesystem>

namespace sgl {

class Resolver {
public:
    virtual ~Resolver() = default;
    virtual std::filesystem::path resolve(const std::filesystem::path& path) = 0;
};

class SearchPathsResolver : public Resolver {
public:
    SearchPathsResolver(std::vector<std::filesystem::path> search_paths)
        : m_search_paths(std::move(search_paths))
    {
    }

    std::filesystem::path resolve(const std::filesystem::path& path) override
    {
        if (!path.is_absolute()) {
            for (const std::filesystem::path& search_path : m_search_paths) {
                auto full_path = search_path / path;
                if (std::filesystem::exists(full_path))
                    return full_path;
            }
        }
        return path;
    }

private:
    std::vector<std::filesystem::path> m_search_paths;
};

} // namespace sgl
