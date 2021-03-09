#pragma once

#include <bitcontainer/bitset.hxx>
#include <bitcontainer/extra/fwd_iterator.hxx>
#include <bitcontainer/sparse_bitset.hxx>

// #include <boost/container/small_vector.hpp>

#include <numeric>
#include <type_traits>
#include <vector>

namespace sd
{

template <typename S>
std::size_t get_dim(const bit_view<S>& s)
{
    return s.empty() ? 0 : (last_entry(s) + 1);
}

template <typename S>
std::size_t get_dim(const sparse_bit_view<S>& s)
{
    return !s.container.empty() ? (s.container.back() + 1) : 0;
}

template <typename S, typename T>
std::size_t size_of_union(const S& s, std::size_t count_s, const T& t, std::size_t count_t)
{
    return count_s + count_t - size_of_intersection(s, t);
}

template <typename S, typename T>
std::size_t size_of_union(const S& s, const T& t)
{
    return size_of_union(s, count(s), t, count(t));
}

// template <typename T, size_t N, typename Alloc = std::allocator<T>>
// using small_vector = boost::container::small_vector<T, N, Alloc>;
// llvm_vecsmall::SmallVector<T, N>;

// template <typename T, size_t N, typename Alloc = std::allocator<T>>
// using small_vector = std::vector<T, Alloc>;

// template <typename T, size_t N, typename Alloc = std::allocator<T>>
// using small_bitset = base_bitset<small_vector<T, N, Alloc>>;
// template <typename T, size_t N, typename Alloc = std::allocator<T>>
// using sparse_small_bitset = resizeable_bitset_sparse<small_vector<T, N, Alloc>>;

namespace disc
{

struct tag_sparse
{
};
struct tag_dense
{
};

template <typename Tag>
constexpr bool is_sparse(Tag&&)
{
    return std::is_same<Tag, tag_sparse>();
}

template <typename Tag>
constexpr bool is_dense(Tag&&)
{
    return std::is_same<Tag, tag_dense>();
}

// using index_type = std::size_t;

#if defined(USE_LONG_INDEX)
using sparse_index_type = std::size_t;
#else
using sparse_index_type = std::uint16_t;
#endif

template <typename T>
using storage_container = std::conditional_t<is_sparse(T{}),
                                             sparse_dynamic_bitset<sparse_index_type>,
                                             dynamic_bitset<std::uint64_t>>;
template <typename T>
using itemset = std::conditional_t<is_sparse(T{}),
                                   sparse_dynamic_bitset<sparse_index_type>,
                                   dynamic_bitset<std::uint64_t>>;
template <typename T>
using long_storage_container = std::conditional_t<is_sparse(T{}),
                                                  sparse_dynamic_bitset<std::uint32_t>,
                                                  dynamic_bitset<std::uint64_t>>;
} // namespace disc

} // namespace sd
