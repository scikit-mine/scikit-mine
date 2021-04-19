#pragma once

#include <bitcontainer/bit_view.hxx>
#include <ndarray/ndarray.hxx>

#include <array>
#include <vector>

namespace sd
{

template <typename T, size_t length>
using static_bitset = sd::bit_view<std::array<T, length>>;

template <typename C, typename = decltype(std::declval<C>().resize(0, 0))>
struct base_bitset : public sd::bit_view<C>
{
    using container_type = C;
    using base           = sd::bit_view<container_type>;
    using block_type     = typename base::block_type;

    explicit base_bitset(size_t highest_bit, bool value = false)
        : base(container_type(base::blocks_needed_to_store_n_bits(highest_bit + 1), 0),
               highest_bit)
    {
        if (value)
        {
            assert(base::count() == 0);
            base::flip_all();
            assert(base::count() == base::length());
        }
    }
    template <typename IterA, typename IterB>
    base_bitset(IterA f, IterB l)
    {
        insert(std::move(f), std::move(l));
    }
    base_bitset() = default;

    base_bitset(slice<const size_t> o) { this->insert(o); }

    // template <typename T, typename = std::enable_if_t<std::is_constructible_v<C, T>>>
    // base_bitset(bit_view<T> const& o) : base{o.container, o.length_}
    // {
    // }
    template <typename T> //, typename = std::enable_if_t<!std::is_constructible_v<C, T>>>
    base_bitset(bit_view<T> const& o)
    {
        this->insert(o);
    }
    // template <typename T>
    // base_bitset& operator=(bit_view<T> const& o)
    // {
    //     this->clear();
    //     this->insert(o);
    //     return *this;
    // }
    // template <typename T, typename = std::enable_if_t<std::is_same_v<T, C>>>
    // base_bitset& operator=(bit_view<T> const& o)
    // {
    //     this->container = o.container;
    //     this->length_ = o.length_;
    //     return *this;
    // }
    // template <typename T, typename = std::enable_if_t<std::is_same_v<T, C>>>
    // base_bitset& operator=(bit_view<T> && o)
    // {
    //     this->container = std::move(o.container);
    //     this->length_ = o.length_;
    //     return *this;
    // }

    void insert(size_t i)
    {
        if (this->length() <= i)
        {
            auto blocks_needed = base::blocks_needed_to_store_n_bits(i + 1);
            this->container.resize(blocks_needed, 0);
            this->length_ = i + 1;
        }
        this->set(i);
    }

    void erase(size_t i)
    {
        if (i < this->length())
            base::unset(i);
    }
    void insert(size_t i, bool value)
    {
        if (value)
            insert(i);
        else
            erase(i);
    }

    template <typename S>
    void insert(const sd::bit_view<S>& rhs)
    {
        this->container.resize(std::max(this->container.size(), rhs.container.size()), 0);
        this->length_ = std::max(this->length(), rhs.length());
        size_t n      = std::min(this->container.size(), rhs.container.size());
        for (size_t i = 0; i < n; ++i)
        {
            this->container[i] |= rhs.container[i];
        }
    }

    template <typename IterA, typename IterB>
    void insert(IterA first, IterB last)
    {
        for (auto it = first; it != last; ++it)
            insert(*it);
    }

    void insert(slice<const size_t> is)
    {
        if (!is.empty())
        {
            reserve(is.back() + 1);
            for (auto i : is.span())
                insert(i);
        }
    }

    template <typename S>
    void assign(S&& rhs)
    {
        clear();
        insert(std::forward<S>(rhs));
    }

    template <typename IterA, typename IterB>
    void assign(IterA first, IterB last)
    {
        clear();
        insert(first, last);
    }

    void reserve(size_t i) { resize(i); }

    void clear()
    { /*this->reset();*/
        this->container.clear();
        this->length_ = 0;
    }
    bool   test(size_t i) const { return is_subset(i, *this); }
    size_t count() const { return base::count(); }

    void resize(size_t n, bool value = false)
    {
        n += 1;
        auto k = base::blocks_needed_to_store_n_bits(n);
        this->container.resize(k, value ? ~block_type(0) : block_type(0));
        this->length_ = n;
    }
};

template <typename S, typename T, typename U>
void intersection(const bit_view<S>& x, const bit_view<T>& y, base_bitset<U>& z)
{
    z.resize(std::max(x.length(), y.length()));
    const auto&  a = x.container;
    const auto&  b = y.container;
    auto&        c = z.container;
    const size_t m = std::min(a.size(), b.size());
    for (size_t i = 0; i < m; ++i)
    {
        c[i] = b[i] & a[i];
    }
    // for (size_t i = m; i < c.size(); ++i)
    // {
    //     c[i] = 0;
    // }
    z.zero_unused_bits();
}

template <typename T, typename Alloc = std::allocator<T>>
using dynamic_bitset = base_bitset<std::vector<T, Alloc>>;

} // namespace sd