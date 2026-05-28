// clang-format off
/* <<<PYMACRO
from itertools import product
for dim in [2, 3]:
    for c in product(["x", "y", "z"], repeat=dim):
        name = "".join(c)
        components = ", ".join(c)
        print(f"[[nodiscard]] constexpr auto {name}() const noexcept {{ return vector<T, {dim}>({components}); }}")
>>> */
[[nodiscard]] constexpr auto xx() const noexcept { return vector<T, 2>(x, x); }
[[nodiscard]] constexpr auto xy() const noexcept { return vector<T, 2>(x, y); }
[[nodiscard]] constexpr auto xz() const noexcept { return vector<T, 2>(x, z); }
[[nodiscard]] constexpr auto yx() const noexcept { return vector<T, 2>(y, x); }
[[nodiscard]] constexpr auto yy() const noexcept { return vector<T, 2>(y, y); }
[[nodiscard]] constexpr auto yz() const noexcept { return vector<T, 2>(y, z); }
[[nodiscard]] constexpr auto zx() const noexcept { return vector<T, 2>(z, x); }
[[nodiscard]] constexpr auto zy() const noexcept { return vector<T, 2>(z, y); }
[[nodiscard]] constexpr auto zz() const noexcept { return vector<T, 2>(z, z); }
[[nodiscard]] constexpr auto xxx() const noexcept { return vector<T, 3>(x, x, x); }
[[nodiscard]] constexpr auto xxy() const noexcept { return vector<T, 3>(x, x, y); }
[[nodiscard]] constexpr auto xxz() const noexcept { return vector<T, 3>(x, x, z); }
[[nodiscard]] constexpr auto xyx() const noexcept { return vector<T, 3>(x, y, x); }
[[nodiscard]] constexpr auto xyy() const noexcept { return vector<T, 3>(x, y, y); }
[[nodiscard]] constexpr auto xyz() const noexcept { return vector<T, 3>(x, y, z); }
[[nodiscard]] constexpr auto xzx() const noexcept { return vector<T, 3>(x, z, x); }
[[nodiscard]] constexpr auto xzy() const noexcept { return vector<T, 3>(x, z, y); }
[[nodiscard]] constexpr auto xzz() const noexcept { return vector<T, 3>(x, z, z); }
[[nodiscard]] constexpr auto yxx() const noexcept { return vector<T, 3>(y, x, x); }
[[nodiscard]] constexpr auto yxy() const noexcept { return vector<T, 3>(y, x, y); }
[[nodiscard]] constexpr auto yxz() const noexcept { return vector<T, 3>(y, x, z); }
[[nodiscard]] constexpr auto yyx() const noexcept { return vector<T, 3>(y, y, x); }
[[nodiscard]] constexpr auto yyy() const noexcept { return vector<T, 3>(y, y, y); }
[[nodiscard]] constexpr auto yyz() const noexcept { return vector<T, 3>(y, y, z); }
[[nodiscard]] constexpr auto yzx() const noexcept { return vector<T, 3>(y, z, x); }
[[nodiscard]] constexpr auto yzy() const noexcept { return vector<T, 3>(y, z, y); }
[[nodiscard]] constexpr auto yzz() const noexcept { return vector<T, 3>(y, z, z); }
[[nodiscard]] constexpr auto zxx() const noexcept { return vector<T, 3>(z, x, x); }
[[nodiscard]] constexpr auto zxy() const noexcept { return vector<T, 3>(z, x, y); }
[[nodiscard]] constexpr auto zxz() const noexcept { return vector<T, 3>(z, x, z); }
[[nodiscard]] constexpr auto zyx() const noexcept { return vector<T, 3>(z, y, x); }
[[nodiscard]] constexpr auto zyy() const noexcept { return vector<T, 3>(z, y, y); }
[[nodiscard]] constexpr auto zyz() const noexcept { return vector<T, 3>(z, y, z); }
[[nodiscard]] constexpr auto zzx() const noexcept { return vector<T, 3>(z, z, x); }
[[nodiscard]] constexpr auto zzy() const noexcept { return vector<T, 3>(z, z, y); }
[[nodiscard]] constexpr auto zzz() const noexcept { return vector<T, 3>(z, z, z); }
/* <<<PYMACROEND>>> */
// clang-format on
