#pragma once

#include "libpopcnt.h"

#include <algorithm> // minmax
#include <cassert>
#include <cmath>      // ceil
#include <functional> // plus
#include <limits>     // char_bit
#include <numeric>    // inner_product

// #include <bit> // std::popcount

namespace sd
{

template <typename Container>
struct bit_view
{
    using container_type = Container;
    using block_type     = typename container_type::value_type;
    using size_type      = typename container_type::size_type;

    static constexpr size_type bytes_per_block = sizeof(block_type);
    static constexpr size_type bits_per_block  = bytes_per_block * __CHAR_BIT__;

    static constexpr size_type blocks_needed_to_store_n_bits(size_type n_bits) noexcept
    {
        return std::ceil(float(n_bits) / bits_per_block);
    }

    bit_view(const bit_view&) = default;
    bit_view(bit_view&&)      = default;
    bit_view& operator=(const bit_view&) = default;
    bit_view& operator=(bit_view&&) = default;

    bit_view() { length_ = num_blocks() * bits_per_block; }

    explicit bit_view(size_type length_) : length_(length_)
    {
        assert(length_ < num_blocks() * bits_per_block);
    }

    bit_view(container_type container, size_type length_)
        : container(std::move(container)), length_(length_)
    {
    }

    explicit bit_view(container_type container)
        : container(std::move(container)), length_(num_blocks() * bits_per_block)
    {
    }

    bool test(size_type k) const
    {
        assert(k < length());
        auto [i, j] = unlinearize(k);
        return container[i] & mask(j);
    }

    bool contains(size_type k) const { return k < length() && test(k); }
    bool operator[](size_type k) const { return contains(k); }

    void set(size_type k)
    {
        assert(k < length());
        auto [i, j] = unlinearize(k);
        container[i] |= mask(j);
    }

    void set(size_type k, bool value)
    {
        if (value)
            set(k);
        else
            unset(k);
    }

    void flip(size_type k)
    {
        assert(k < length());
        auto [i, j] = unlinearize(k);
        container[i] ^= mask(j);
    }

    void unset(size_type k)
    {
        assert(k < length());
        auto [i, j] = unlinearize(k);
        container[i] &= ~mask(j);
    }

    void reset() { std::fill_n(container.begin(), container.size(), block_type()); }
    void clear() { reset(); }

    size_type count() const { return popcnt(container.data(), bytes_per_block * num_blocks()); }
    size_type length() const { return length_; }
    bool      empty() const { return length() == 0 || count() == 0; }
    size_type num_blocks() const { return container.size(); }

    template <typename S>
    auto& operator&=(const bit_view<S>& rhs)
    {
        auto n = std::min(this->container.size(), rhs.container.size());
        auto m = this->container.size();
        for (size_t i = 0; i < n; ++i)
        {
            this->container[i] &= rhs.container[i];
        }
        for (size_t i = n; i < m; ++i)
        {
            this->container[i] = 0;
        }
        return *this;
    }

    template <typename S>
    auto& operator|=(const bit_view<S>& rhs)
    {
        auto n = std::min(this->container.size(), rhs.container.size());
        for (size_t i = 0; i < n; ++i)
        {
            this->container[i] |= rhs.container[i];
        }
        zero_unused_bits();
        return *this;
    }

    template <typename S>
    auto& operator^=(const bit_view<S>& rhs)
    {
        auto n = std::min(this->container.size(), rhs.container.size());
        for (size_t i = 0; i < n; ++i)
        {
            this->container[i] ^= rhs.container[i];
        }
        zero_unused_bits();
        return *this;
    }

    size_t get_unused_bits() const
    {
        auto [i, j] = unlinearize(length());
        return j;
    }

    void zero_unused_bits()
    {
        auto extra_bits = get_unused_bits();
        if (extra_bits != 0)
            container.back() &= ~(~static_cast<block_type>(0) << extra_bits);
    }

    void flip_all()
    {
        for (size_t k = 0; k < container.size(); ++k)
        {
            container[k] = ~container[k];
        }
        zero_unused_bits();
    }

    size_t highest_used_block() const { return unlinearize(length()).first; }

    constexpr std::pair<size_type, size_type> unlinearize(size_type i) const noexcept
    {
        return {i / bits_per_block, i % bits_per_block};
    }
    constexpr block_type mask(size_type i) const noexcept { return block_type(1) << i; }

    container_type container = {};
    size_type      length_   = 0;
};

template <typename S>
void swap(bit_view<S>& a, bit_view<S>& b)
{
    using std::swap;

    swap(a.container, b.container);
    swap(a.length_, b.length_);
}

template <typename S, typename T>
auto operator&(const bit_view<S>& lhs, const bit_view<S>& rhs)
{
    auto r = lhs;
    r &= rhs;
    return r;
}

template <typename S, typename T>
auto operator|(const bit_view<S>& lhs, const bit_view<S>& rhs)
{
    auto r = lhs;
    r |= rhs;
    return r;
}

template <typename S, typename T>
auto operator^(const bit_view<S>& lhs, const bit_view<S>& rhs)
{
    auto r = lhs;
    r ^= rhs;
    return r;
}

template <typename T>
std::size_t most_significant_bit(T x)
{
    // return __builtin_clzll(x);
    std::size_t r = 0;
    if (x < 1)
        return 0;
    while (x >>= T(1))
    {
        ++r;
    }
    return r;
}

template <typename T>
std::size_t least_significant_bit(T x)
{
    // return __builtin_ffs(x);
    std::size_t r = 0;
    while (!(x & 1))
    {
        x >>= 1;
        ++r;
    }
    return r;
}

template <typename S>
std::size_t last_entry(const bit_view<S>& s)
{
    constexpr auto bits_per_block = bit_view<S>::bits_per_block;
    const auto&    c              = s.container;
    for (size_t i = c.size(); i-- > 0;)
    {
        if (c[i] != 0)
        {
            return (bits_per_block * i) + most_significant_bit(c[i]);
        }
    }
    return 0;
}

template <typename S, typename T>
size_t largest_common_block(const bit_view<S>& s, const bit_view<T>& t)
{
    return std::min(s.container.size(), t.container.size());
    // return std::min({s.container.size(), t.container.size(), s.highest_used_block() + 1,
    // t.highest_used_block() + 1});
}

template <typename S, typename T>
bool is_subset(const bit_view<S>& s, const bit_view<T>& t)
{
    if (s.length() == 0)
        return true; // s is empty set

    // do not call empty() to prevent the second call to count()

    const auto cnt_s = s.count();
    if (cnt_s == 0)
        return true; // s is empty set

    if (cnt_s > t.count())
        return false;

    const auto& a = s.container;
    const auto& b = t.container;
    for (size_t i = 0, n = largest_common_block(s, t); i < n; ++i)
    {
        if ((a[i] & ~b[i]))
            return false;
        // if ( (a[i] & b[i]) != a[i]) { return false; }
    }

    if (a.size() <= b.size())
    {
        return true;
    }
    else
    {
        return std::all_of(a.begin() + b.size(), a.end(), [](const auto& x) { return x == 0; });
    }
}

template <typename S>
bool is_subset(size_t i, const bit_view<S>& x)
{
    return i < x.length() && x.contains(i);
}

template <typename S, typename T>
bool is_proper_subset(const bit_view<S>& s, const bit_view<T>& t)
{
    return s.count() < t.count() && is_subset(s, t);
}

template <typename S, typename T>
bool intersects(const bit_view<S>& s, const bit_view<T>& t)
{
    const auto& a = s.container;
    const auto& b = t.container;
    for (size_t i = 0, n = largest_common_block(s, t); i < n; ++i)
    {
        if ((a[i] & b[i]))
            return true;
    }
    return false;
}

template <typename S, typename T>
void intersection(const bit_view<S>& s, bit_view<T>& t)
{
    const auto&  a = s.container;
    auto&        b = t.container;
    const size_t m = largest_common_block(s, t);
    for (size_t i = 0; i < m; ++i)
    {
        b[i] &= a[i];
    }
    for (size_t i = m; i < b.size(); ++i)
    {
        b[i] = 0;
    }
}

template <typename S, typename T>
auto size_of_intersection(const bit_view<S>& s, const bit_view<T>& t) -> uint64_t
{
    const auto& a = s.container;
    const auto& b = t.container;

    static_assert(std::is_convertible_v<std::decay_t<decltype((a[0] & b[0]))>, uint64_t>, "");

    if (a.size() > b.size())
    {
        return size_of_intersection(t, s);
    }

    return std::inner_product(a.begin(),
                              a.end(),
                              b.begin(),
                              uint64_t(),
                              std::plus<>{},
                              [](const auto& x, const auto& y) {
                                  return popcnt64(uint64_t{x & y});
                                  //   std::decay_t<decltype(x & y)> z[1]  = {x & y};
                                  //   return popcnt(z, sizeof(z[0]));
                              });
}

template <typename S, typename T>
bool equal(const bit_view<S>& s, const bit_view<T>& t)
{
    const auto& a = s.container;
    const auto& b = t.container;

    for (size_t i = 0, n = largest_common_block(s, t); i < n; ++i)
    {
        if ((a[i] != b[i]))
            return false;
    }

    if (a.size() == b.size())
    {
        return true;
    }
    else if (a.size() > b.size())
    {
        return std::all_of(a.begin() + b.size(), a.end(), [](const auto& x) { return x == 0; });
    }
    else
    {
        return std::all_of(b.begin() + a.size(), b.end(), [](const auto& x) { return x == 0; });
    }
}

// template <typename S, typename Fn>
// void iterate_over_old_and_crappy(const bit_view<S>& s, Fn&& fn)
// {
//     for (size_t i = 0; i < s.length(); ++i)
//     {
//         if (s.contains(i))
//         {
//             fn(i);
//         }
//     }
// }

template <typename S, typename Fn>
void foreach(const bit_view<S>& s, Fn&& fn)
{
    size_t offset = 0;
    for (auto x : s.container)
    {
        for (size_t i = 0; x; ++i)
        {
            if ((x & 1))
            {
                // co_yield i + offset;
                fn(i + offset);
            }
            x >>= 1;
        }
        offset += bit_view<S>::bits_per_block;
        if (offset >= s.length())
            break;
    }
}

// s <- s \ t
template <typename S, typename T>
void setminus(bit_view<S>& s, const bit_view<T>& t)
{
    const auto& a  = t.container;
    auto&       b  = s.container;
    const auto  ub = largest_common_block(s, t);
    for (size_t i = 0; i < ub; ++i)
    {
        b[i] &= ~(a[i] & b[i]);
    }
}

// u <- s \ t
template <typename S, typename T, typename U>
void setminus(const bit_view<S>& s, const bit_view<T>& t, bit_view<U>& u)
{
    const auto& a  = t.container;
    const auto& b  = s.container;
    auto&       c  = u.container;
    const auto  ub = std::min({a.size(), b.size(), c.size()});
    for (size_t i = 0; i < ub; ++i)
    {
        c[i] = b[i] & ~(a[i] & b[i]);
    }
    // insert remaining parts from `s`
    const auto ub1 = std::min(b.size(), c.size());
    for (size_t i = ub; i < ub1; ++i)
    {
        c[i] = b[i];
    }
    // clear rest
    for (size_t i = ub1; i < c.size(); ++i)
    {
        c[i] = 0;
    }
}

template <typename S>
size_t count(const bit_view<S>& s)
{
    return s.count();
}

template <typename S>
bool is_singleton(const bit_view<S>& s)
{
    return count(s) == 1;
}

template <typename S>
size_t front(const bit_view<S>& s)
{
    assert(count(s) != 0);

    constexpr auto bpb = bit_view<S>::bits_per_block;

    for (size_t i = 0; i < s.container.size(); ++i)
    {
        if (s.container[i] != 0)
        {
            return (bpb * i) + least_significant_bit(s.container[i]);
        }
    }

    assert(false);
    return size_t(-1);
}

template <typename S>
bool none_of(const bit_view<S>& s)
{
    for (size_t i = 0, n = s.container.size(); i < n; ++i)
    {
        if (s.container[i] != 0)
        {
            return false;
        }
    }
    return true;
}

template <typename S>
bool any_of(const bit_view<S>& s)
{
    return !none_of(s);
}

template <typename S>
bool all_of(const bit_view<S>& s)
{
    return count(s) == s.length();
}

} // namespace sd