#pragma once

#include <ndarray/ndarray.hxx> // slice

#include "meta.hxx"
#include "zip_iterator.hxx"

#include <memory>
#include <tuple>
#include <utility>

namespace sd
{

template <typename T>
auto make_slice(T& t)
{
    // this exists only to make clang happy, gcc can deduce the proper type
    //  o/w sed 's/sd::make_slice(/sd::slice(/g' -i disc/datatable/data_table.hxx
    //  desc/storage/Dataset.hxx relaxent/IntersectingFacotorization.hxx
    return sd::slice<std::remove_reference_t<decltype(*t.data())>, 1>(t.data(), t.size());
}

template <typename T>
auto make_cpslice(T& t)
{
    return sd::cpslice<std::remove_reference_t<decltype(*t.data())>, 1>(t.data(), t.size());
}

namespace df
{

template <typename... Ts>
struct basic_soa;

template <typename... Ts>
struct basic_soa
{
    using column_tuple = std::tuple<Ts...>;

    column_tuple columns;

    basic_soa()                 = default;
    basic_soa(const basic_soa&) = default;
    basic_soa(basic_soa&&)      = default;
    basic_soa& operator=(const basic_soa&) = default;
    basic_soa& operator=(basic_soa&&) = default;

    template <typename... Us>
    explicit basic_soa(basic_soa<Us...>& x)
    {
        columns = x.columns;
    }

    template <typename... Us>
    explicit basic_soa(const basic_soa<Us...>& x)
    {
        columns = x.columns;
    }

    template <typename... Us>
    explicit basic_soa(basic_soa<Us...>&& x)
    {
        columns = std::forward<basic_soa<Us...>>(x.columns);
    }

    template <typename... Us>
    explicit basic_soa(std::tuple<Us...>&& t) : columns(std::forward<std::tuple<Us...>>(t))
    {
    }

    template <typename... Us>
    explicit basic_soa(Us&&... t) : columns(std::forward_as_tuple<Us...>(t...))
    {
    }

    size_t size() const { return sizeof...(Ts) == 0 ? 0 : std::get<0>(columns).size(); }
    size_t empty() const { return size() == 0; }

    constexpr static size_t num_columns() { return sizeof...(Ts); }

    auto begin() const { return sd::zip(columns).begin(); }
    auto end() const { return sd::zip(columns).end(); }
    auto begin() { return sd::zip(columns).begin(); }
    auto end() { return sd::zip(columns).end(); }

    auto rbegin() const { return std::make_reverse_iterator(end()); }
    auto rend() const { return std::make_reverse_iterator(begin()); }
    auto rbegin() { return std::make_reverse_iterator(end()); }
    auto rend() { return std::make_reverse_iterator(begin()); }

    template <typename Fn>
    void foreach_col(Fn&& fn) const
    {
        foreach_tuple(columns, std::forward<Fn>(fn));
    }
    template <typename Fn>
    void foreach_col(Fn&& fn)
    {
        foreach_tuple(columns, std::forward<Fn>(fn));
    }

    template <typename Fn>
    auto map_cols(Fn&& fn) const
    {
        return map_cols_impl(std::forward<Fn>(fn), std::index_sequence_for<Ts...>());
    }
    template <typename Fn>
    auto map_cols(Fn&& fn)
    {
        return map_cols_impl(std::forward<Fn>(fn), std::index_sequence_for<Ts...>());
    }

    template <size_t... Is>
    auto iter()
    {
        return zip(std::get<Is>(columns)...);
    }
    template <size_t... Is>
    auto iter() const
    {
        return zip(std::get<Is>(columns)...);
    }

    template <typename Fn>
    auto map_cols_unwrapped(Fn&& fn) const
    {
        return map_tuple(columns, std::forward<Fn>(fn));
    }
    template <typename Fn>
    auto map_cols_unwrapped(Fn&& fn)
    {
        return map_tuple(columns, std::forward<Fn>(fn));
    }

    template <size_t index>
    decltype(auto) col()
    {
        return std::get<index>(columns);
    }
    template <size_t index>
    decltype(auto) col() const
    {
        return std::get<index>(columns);
    }
    auto operator[](size_t row_index)
    {
        return get(row_index, std::index_sequence_for<Ts...>());
    }
    auto operator[](size_t row_index) const
    {
        return get(row_index, std::index_sequence_for<Ts...>());
    }
    decltype(auto) back() { return this->operator[](size() - 1); }
    decltype(auto) back() const { return this->operator[](size() - 1); }

private:
    template <typename Fn, size_t... Is>
    auto map_cols_impl(Fn&& fn, std::index_sequence<Is...>&&)
    {
        using result_type = basic_soa<std::decay_t<decltype(fn(std::get<Is>(columns)))>...>;
        return result_type(map_tuple(columns, std::forward<Fn>(fn)));
    }

    template <typename Fn, size_t... Is>
    auto map_cols_impl(Fn&& fn, std::index_sequence<Is...>&&) const
    {
        using result_type = basic_soa<std::decay_t<decltype(fn(std::get<Is>(columns)))>...>;
        return result_type(map_tuple(columns, std::forward<Fn>(fn)));
    }

    template <size_t... Is>
    auto get(size_t i, std::index_sequence<Is...>&&)
    {
        return std::tie(std::get<Is>(columns)[i]...);
    }
    template <size_t... Is>
    auto get(size_t i, std::index_sequence<Is...>&&) const
    {
        return std::tie(std::get<Is>(columns)[i]...);
    }
};

template <typename... Ts>
struct basic_column_store;

template <typename... Ts>
struct basic_column_store : public basic_soa<Ts...>
{
    using basic_soa<Ts...>::basic_soa;

    template <typename... Args>
    auto cut(Args&&... args)
    {
        return this->map_cols(
            [&](auto& col) { return sd::make_cpslice(col).cut(std::forward<Args>(args)...); });
    }

    template <typename... Args>
    auto cut(Args&&... args) const
    {
        return this->map_cols([&](const auto& col) {
            return sd::make_slice(col).cut(std::forward<Args>(args)...);
        });
    }

    template <typename... Args>
    auto stride(Args&&... args)
    {
        return this->map_cols([&](auto& col) {
            return sd::make_cpslice(col).stride(std::forward<Args>(args)...);
        });
    }

    template <typename... Args>
    auto stride(Args&&... args) const
    {
        return this->map_cols([&](const auto& col) {
            return sd::make_slice(col).stride(std::forward<Args>(args)...);
        });
    }

    template <size_t... Is>
    auto select_columns()
    {
        return basic_column_store(sd::make_cpslice(std::get<Is>(this->columns))...);
    }

    template <size_t... Is>
    auto select_columns() const
    {
        return basic_column_store(sd::make_slice(std::get<Is>(this->columns)...));
    }

    void reserve(size_t n)
    {
        this->foreach_col([n](auto& c) { return c.reserve(n); });
    }

    void resize(size_t n)
    {
        this->foreach_col([n](auto& c) { return c.resize(n); });
    }

    void shrink_to_fit()
    {
        this->foreach_col([](auto& c) { return c.shrink_to_fit(); });
    }

    void clear()
    {
        this->foreach_col([](auto& c) { return c.clear(); });
    }

    size_t size() const { return sizeof...(Ts) == 0 ? 0 : std::get<0>(this->columns).size(); }
    size_t empty() const { return size() == 0; }

    template <typename... Us>
    void push_back(Us&&... us)
    {
        push_back_impl(std::forward_as_tuple(us...), std::index_sequence_for<Us...>());
    }
    template <typename... Us>
    void push_back(std::tuple<Us...>&& t)
    {
        push_back_impl(std::forward<std::tuple<Us...>>(t), std::index_sequence_for<Us...>());
    }

    template <typename... Us>
    decltype(auto) emplace_back(Us&&... us)
    {
        emplace_back_impl(std::forward_as_tuple(us...), std::index_sequence_for<Us...>());
        return this->back();
    }
    template <typename... Us>
    decltype(auto) emplace_back(std::tuple<Us...>&& t)
    {
        emplace_back_impl(std::forward<std::tuple<Us...>>(t), std::index_sequence_for<Us...>());
        return this->back();
    }

    void pop_back()
    {
        this->foreach_col([](auto& c) { return c.pop_back(); });
    }

    template <typename Fn>
    auto map_cols(Fn&& fn) const
    {
        return map_wrapped_impl(std::forward<Fn>(fn), std::index_sequence_for<Ts...>());
    }
    template <typename Fn>
    auto map_cols(Fn&& fn)
    {
        return map_wrapped_impl(std::forward<Fn>(fn), std::index_sequence_for<Ts...>());
    }

    void erase_row(size_t row)
    {
        this->foreach_col([row](auto& c) { return c.erase(c.begin() + row); });
    }

    void erase_rows(size_t from, size_t to)
    {
        this->foreach_col(
            [from, to](auto& c) { return c.erase(c.begin() + from, c.begin() + to); });
    }

    template <typename Pred>
    void erase_if(Pred&& pred)
    {
        auto it = std::remove_if(this->begin(), this->end(), std::forward<Pred>(pred));
        auto d  = std::distance(it, this->end());
        this->foreach_col([d](auto& c) { return c.erase(c.begin() + d, c.end()); });
    }

private:
    template <typename tuple_t, size_t... Is>
    void push_back_impl(tuple_t&& t, std::index_sequence<Is...>&&)
    {
        [[maybe_unused]] auto _ = std::initializer_list<int>{
            (std::get<Is>(this->columns).push_back(std::get<Is>(t)), 0)...};
    }
    template <typename tuple_t, size_t... Is>
    void emplace_back_impl(tuple_t&& t, std::index_sequence<Is...>&&)
    {
        [[maybe_unused]] auto _ = std::initializer_list<int>{
            (std::get<Is>(this->columns).emplace_back(std::get<Is>(t)), 0)...};
    }
    template <typename Fn, size_t... Is>
    auto map_wrapped_impl(Fn&& fn, std::index_sequence<Is...>&&)
    {
        using result_type =
            basic_column_store<std::decay_t<decltype(fn(std::get<Is>(this->columns)))>...>;
        return result_type(map_tuple(this->columns, std::forward<Fn>(fn)));
    }

    template <typename Fn, size_t... Is>
    auto map_wrapped_impl(Fn&& fn, std::index_sequence<Is...>&&) const
    {
        using result_type =
            basic_column_store<std::decay_t<decltype(fn(std::get<Is>(this->columns)))>...>;
        return result_type(map_tuple(this->columns, std::forward<Fn>(fn)));
    }
};

template <typename T>
struct column_type
{
    using value = std::vector<T>;
};

template <typename... Ts>
using col_store = basic_column_store<typename column_type<Ts>::value...>;

template <typename... Ts, std::size_t... Is>
void tuple_swap_impl(std::tuple<Ts&...>&& a,
                     std::tuple<Ts&...>&& b,
                     std::index_sequence<Is...>&&)
{
    using std::swap;
    [[maybe_unused]] auto _ =
        std::initializer_list<int>{(swap(std::get<Is>(a), std::get<Is>(b)), 0)...};
}

} // namespace df
} // namespace sd

namespace std
{
template <typename... Ts>
void swap(std::tuple<Ts&...>&& a, std::tuple<Ts&...>&& b)
{
    sd::df::tuple_swap_impl(std::forward<std::tuple<Ts&...>>(a),
                            std::forward<std::tuple<Ts&...>>(b),
                            std::index_sequence_for<Ts...>());
}
} // namespace std

// template <typename T, size_t r = 1>
// struct column_type : public ndarray<T, r>
// {
//     using typename ndarray<T, r>::value_type;
//     using ndarray<T, r>::ndarray;
// };

// template <typename T, size_t r>
// struct col
// {
// };

// template <typename T, size_t r>
// struct column_type<col<T, r>>
// {
//     using value = ndarray<T, r>;
// };

// template <typename... Ts>
// using soa_vector = basic_soa<column_type<Ts>...>;

// template <typename... Ts>
// using soa_view = basic_soa<cpslice<Ts>...>;

// template <typename... Ts>
// soa_view<Ts...> view(basic_soa<column_type<Ts>...>& d)
// {
//     return soa_view<Ts...>(d);
// }

// template <typename T, typename... Ts>
// class row_store : public ndarray<std::tuple<T, Ts...>, 1, std::allocator<std::tuple<T,
// Ts...>>>
// {
//     using base = ndarray<std::tuple<T, Ts...>, 1, std::allocator<std::tuple<T, Ts...>>>;

// public:
//     using base::base;

//     void push_back(T&& t0, Ts&&... ts)
//     {
//         base::push_back(std::forward_as_tuple(std::forward<T>(t0), std::forward<Ts>(ts)...));
//     }
// };

// template <typename T, size_t R, typename Alloc = std::allocator<T>>
// struct data_grid : public ndarray<T, R, Alloc>
// {
//     using range       = row_range<T, R>;
//     using const_range = row_range<const T, R>;

//     using ndarray<T, R, Alloc>::ndarray;

//     auto begin() { range{*this}.begin(); }
//     auto end() { range{*this}.end(); }
//     auto begin() const { const_range{*this}.begin(); }
//     auto end() const { const_range{*this}.end(); }
//     auto rbegin() { range{*this}.rbegin(); }
//     auto rend() { range{*this}.rend(); }
//     auto rbegin() const { const_range{*this}.rbegin(); }
//     auto rend() const { const_range{*this}.rend(); }
// };
