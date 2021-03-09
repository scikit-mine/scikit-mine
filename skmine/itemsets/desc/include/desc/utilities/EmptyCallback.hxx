#pragma once

namespace sd
{

struct EmptyCallback
{
    template <typename... Args>
    inline constexpr void operator()(Args&&... /*unsused*/) const noexcept
    {
    }
};

} // namespace sd