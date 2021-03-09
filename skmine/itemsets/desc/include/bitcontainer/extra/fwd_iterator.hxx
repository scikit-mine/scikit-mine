#pragma once

#include <bitcontainer/bitset.hxx>
#include <bitcontainer/sparse_bitset.hxx>

namespace sd
{

struct bit_view_fwd_iterator_sentinel
{
};

template <typename S>
struct bit_view_fwd_iterator
{
    bit_view<S> const* s;
    size_t             x     = 0;
    size_t             index = 0;
    size_t             pos   = 0;

    bit_view_fwd_iterator(std::nullptr_t) : s(nullptr) {}

    bit_view_fwd_iterator(bit_view<S> const* s) : s(s)
    {
        if (s->container.size() > pos)
        {
            x = s->container[pos++];
            seek();
        }
        else
        {
            s = nullptr;
        }
    }

    size_t operator*() const { return index - 1; }

    auto operator++()
    {
        seek();
        return *this;
    }

    void seek()
    {
        if (!s)
            return;
        while (x == 0)
        {
            if (pos < s->container.size())
            {
                x = s->container[pos++];
            }
            else
            {
                s = nullptr;
                return;
            }
        }

        while (x != 0 && s)
        {
            if ((x & 1))
            {
                index++;
                x >>= 1;
                return;
            }
            index++;
            x >>= 1;
        }
    }
    bool operator==(bit_view_fwd_iterator_sentinel) const { return !s; }
    bool operator!=(bit_view_fwd_iterator_sentinel end) const { return !(*this == end); }
};

template <typename S>
bit_view_fwd_iterator<S> begin(const bit_view<S>& s)
{
    if (s.length() == 0)
        return {nullptr};
    else
        return bit_view_fwd_iterator<S>(&s);
}
template <typename S>
bit_view_fwd_iterator_sentinel end(const bit_view<S>&)
{
    return {};
}

// template <typename S>
// bit_view_fwd_iterator<S> begin(base_bitset<S>& s)
// {
//     if (s.length() == 0)
//         return {nullptr};
//     else
//         return bit_view_fwd_iterator<S>(&s);
// }
// template <typename S>
// bit_view_fwd_iterator_sentinel end(base_bitset<S>&)
// {
//     return {};
// }

template <typename S>
decltype(auto) begin(const sparse_bit_view<S>& s)
{
    using std::begin;
    return begin(s.container);
}

template <typename S>
decltype(auto) end(const sparse_bit_view<S>& s)
{
    using std::end;
    return end(s.container);
}

} // namespace sd