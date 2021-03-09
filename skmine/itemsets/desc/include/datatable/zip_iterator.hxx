#pragma once
#include <iterator>
#include <tuple>
#include <utility>

#include "meta.hxx"

namespace sd
{
namespace impl
{

template <typename Reference>
struct arrow_proxy
{
    Reference  r;
    Reference* operator->() { return &r; }
};

template <typename... Iters>
class zip_iterator
{
public:
    using value_type        = std::tuple<typename std::iterator_traits<Iters>::value_type...>;
    using difference_type   = std::ptrdiff_t;
    using reference         = value_type&;
    using pointer           = value_type*;
    using iterator_category = std::forward_iterator_tag;

    zip_iterator(Iters&&... iters) : zipped(iters...) {}

    auto operator*()
    {
        return map_tuple(
            zipped, [](auto& iter) -> auto& { return *iter; });
    }
    auto operator*() const
    {
        return map_tuple(
            zipped, [](const auto& iter) -> const auto& { return *iter; });
    }

    auto operator->() const { return arrow_proxy(this->operator*()); }

    zip_iterator& operator++()
    {
        foreach_tuple(zipped, [](auto& iter) { ++iter; });
        return *this;
    }

    bool operator==(const zip_iterator& rhs) const { return zipped == rhs.zipped; }
    bool operator!=(const zip_iterator& rhs) const { return !(*this == rhs); }

    // additional interface
    auto& operator--()
    {
        foreach_tuple(zipped, [](auto& iter) { --iter; });
        return *this;
    }
    zip_iterator operator+(const zip_iterator& rhs) const { return zipped + rhs.zipped; }
    zip_iterator operator+(std::ptrdiff_t n) const
    {
        auto it = *this;
        it += n;
        return it;
    }
    zip_iterator operator-(std::ptrdiff_t n) const
    {
        auto it = *this;
        it -= n;
        return it;
    }
    std::ptrdiff_t operator-(const zip_iterator& rhs) const
    {
        return std::get<0>(zipped) - std::get<0>(rhs.zipped);
    }

    zip_iterator& operator+=(std::ptrdiff_t n)
    {
        foreach_tuple(zipped, [n](auto& i) { i += n; });
        return *this;
    }
    zip_iterator& operator-=(std::ptrdiff_t n)
    {
        foreach_tuple(zipped, [n](auto& i) { i -= n; });
        return *this;
    }
    zip_iterator operator++(int)
    {
        auto cpy = *this;
        foreach_tuple(cpy.zipped, [](auto& i) { i++; });
        return cpy;
    }
    zip_iterator operator--(int)
    {
        auto cpy = *this;
        foreach_tuple(cpy.zipped, [](auto& i) { i--; });
        return cpy;
    }

    bool operator<(const zip_iterator& rhs) const { return zipped < rhs.zipped; }
    bool operator<=(const zip_iterator& rhs) const { return zipped <= rhs.zipped; }
    bool operator>(const zip_iterator& rhs) const { return zipped > rhs.zipped; }
    bool operator>=(const zip_iterator& rhs) const { return zipped >= rhs.zipped; }

private:
    std::tuple<Iters...> zipped;
};

} // namespace impl

template <typename Iter>
class zipped_range
{
public:
    using iterator = Iter;
    zipped_range(Iter begin, Iter end) : begin_(std::move(begin)), end_(std::move(end)) {}
    iterator begin() const { return begin_; }
    iterator end() const { return end_; }

private:
    Iter begin_, end_;
};

template <typename... Iter>
impl::zip_iterator<Iter...> make_zipped_iterator(Iter... iter)
{
    return {std::move(iter)...};
}

template <typename Iter>
zipped_range<Iter> make_zipped_range(Iter first, Iter last)
{
    return {std::move(first), std::move(last)};
}

template <typename... Seqs>
auto zip(Seqs&&... seqs)
{
    return make_zipped_range(make_zipped_iterator(seqs.begin()...),
                             make_zipped_iterator(seqs.end()...));
}

template <typename... Seqs, std::size_t... Is>
auto zip(std::tuple<Seqs...>& seqs, std::index_sequence<Is...>&&)
{
    return make_zipped_range(make_zipped_iterator(std::get<Is>(seqs).begin()...),
                             make_zipped_iterator(std::get<Is>(seqs).end()...));
}

template <typename... Seqs, std::size_t... Is>
auto zip(const std::tuple<Seqs...>& seqs, std::index_sequence<Is...>&&)
{
    return make_zipped_range(make_zipped_iterator(std::get<Is>(seqs).begin()...),
                             make_zipped_iterator(std::get<Is>(seqs).end()...));
}

template <typename... Seqs>
auto zip(std::tuple<Seqs...>& seqs)
{
    return zip(seqs, std::index_sequence_for<Seqs...>());
}

template <typename... Seqs>
auto zip(const std::tuple<Seqs...>& seqs)
{
    return zip(seqs, std::index_sequence_for<Seqs...>());
}

} // namespace sd
