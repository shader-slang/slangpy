// clang-format off
/* <<<PYMACRO
from itertools import product
for c in product(["x", "y"], repeat=2):
    name = "".join(c)
    components = ", ".join(c)
    print(f"[[nodiscard]] constexpr auto {name}() const noexcept {{ return vector<T, 2>({components}); }}")
>>> */
[[nodiscard]] constexpr auto xx() const noexcept { return vector<T, 2>(x, x); }
[[nodiscard]] constexpr auto xy() const noexcept { return vector<T, 2>(x, y); }
[[nodiscard]] constexpr auto yx() const noexcept { return vector<T, 2>(y, x); }
[[nodiscard]] constexpr auto yy() const noexcept { return vector<T, 2>(y, y); }
/* <<<PYMACROEND>>> */
// clang-format on
