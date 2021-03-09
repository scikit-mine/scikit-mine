#pragma once

#include <unordered_map>

namespace sd::disc
{

template <typename From>
struct BimapImpl
{
    std::unordered_map<From, size_t> to_internal;
    std::unordered_map<size_t, From> to_external;
    size_t                           next_index = 0;

    bool empty() const { return to_internal.empty(); }

    size_t convert_to_internal(const From& x)
    {
        auto p = to_internal.find(x);
        if (p == to_internal.end())
        {
            to_internal.emplace(x, next_index);
            to_external.emplace(next_index, x);
            next_index = next_index + 1;
            return next_index - 1;
        }
        return p->second;
    }

    size_t convert_to_external(size_t index) const { return to_external.at(index); }
};

using BiMap = BimapImpl<size_t>;

} // namespace sd::disc
