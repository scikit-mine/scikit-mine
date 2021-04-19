#pragma once

#include <tuple>
#include <utility>

namespace sd
{

template <typename T, typename F, std::size_t... Is>
constexpr void foreach_tuple_impl(T& t, F& f, std::index_sequence<Is...>&&)
{
    using std::get;
    [[maybe_unused]] auto dummy = {(f(get<Is>(t)), 0)...};
}

template <typename F, typename... Ts>
constexpr void foreach_tuple(std::tuple<Ts...>& t, F&& f)
{
    foreach_tuple_impl(t, f, std::index_sequence_for<Ts...>());
}

template <typename T, typename F, std::size_t... Is>
constexpr auto map_tuple_impl(T&& t, F&& f, std::index_sequence<Is...>&&)
{
    using std::get;
    return std::tuple<decltype((f(get<Is>(t))))...>(f(get<Is>(t))...);
}

template <typename F, typename... Ts>
constexpr auto map_tuple(std::tuple<Ts...>& t, F&& f)
{
    return map_tuple_impl(t, std::forward<F>(f), std::index_sequence_for<Ts...>());
}

template <typename F, typename... Ts>
constexpr auto map_tuple(const std::tuple<Ts...>& t, F&& f)
{
    return map_tuple_impl(t, std::forward<F>(f), std::index_sequence_for<Ts...>());
}

} // namespace sd
