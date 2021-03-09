#pragma once

#include <desc/Desc.hxx>

namespace sd::disc
{

constexpr size_t iterated_log2(double n)
{
    // iterated log
    if (n <= 1) return 0;
    if (n <= 2) return 1;
    if (n <= 4) return 2;
    if (n <= 16) return 3;
    if (n <= 65536) return 4;
    return 5; // if (n <= 2^65536)
    // never 6 for our case.
}

constexpr double universal_code(size_t n) { return iterated_log2(n) + 1.5186; }
// 1.5186 = log2(2.865064)

template <typename T = double, typename X>
auto constant_mdl_cost(Component<T> const& c, const X& x)
{
    typename T::float_type acc = 0;
    // each item in pattern:
    foreach (x, [&](size_t item) {
        auto s = c.frequency[item] * c.data.size();
        auto a = s == 0;
        acc -= std::log2((s + a) / (c.data.size() + a));
    })
        ;
    return acc + universal_code(count(x));
}

template <typename T, typename X>
auto constant_mdl_cost(const Composition<T>& c, const X& x)
{
    typename T::float_type acc = 0;

    foreach (x, [&](size_t i) {
        auto s = c.frequency(i, 0) * c.data.subset(0).size();
        auto a = s == 0;
        acc -= std::log2((s + a) / (c.data.subset(0).size() + a));
    })
        ;
    acc += universal_code(count(x));

    return acc;
}
auto additional_cost_mdl(size_t support)
{
    return universal_code(support); // encode-per-component-support
}

template <typename Trait, typename Candidate>
auto desc_heuristic_mdl_multi(const Composition<Trait>& c, const Candidate& x)
{
    using float_type = typename Trait::float_type;

    float_type acc = -constant_mdl_cost(c, x.pattern);

    for (size_t i = 0; i < c.data.num_components(); ++i)
    {
        auto p = c.models[i].expectation(x.pattern);
        auto s = size_of_intersection(x.row_ids, c.masks[i]);
        auto q = static_cast<float_type>(s) / c.data.subset(i).size();
        auto h = s == 0 ? 0 : s * std::log2(q / p);

        assert(0 <= p && p <= 1);

        acc += h - additional_cost_mdl(s);
    }

    return acc;
}

template <typename C, typename Distribution, typename Candidate>
auto desc_heuristic_mdl_1(const C& c, const Distribution& pr, const Candidate& x)
{
    using float_type = typename C::float_type;

    const auto s = static_cast<float_type>(x.support);
    const auto q = s / c.data.size();
    const auto p = pr.expectation(x.pattern);

    assert(0 <= p && p <= 1);

    return s * std::log2(q / p) - constant_mdl_cost(c, x.pattern) - additional_cost_mdl(x.support);
}

struct IDescMDL : DefaultPatternsetMinerInterface
{
    template <typename T, typename Candidate, typename Config>
    static auto heuristic(Component<T>& c, Candidate& x, const Config&)
    {
        return desc_heuristic_mdl_1(c, c.model, x);
    }

    template <typename T, typename Candidate, typename Config>
    static auto heuristic(Composition<T>& c, Candidate& x, const Config&)
    {
        if (c.data.num_components() == 1)
        {
            return desc_heuristic_mdl_1(c, c.models.front(), x);
        }
        else
        {
            return desc_heuristic_mdl_multi(c, x);
        }
    }

    template <typename C, typename Config>
    static auto finish(C& c, const Config& cfg)
    {
        prune_model(c, cfg);
    }
};

} // namespace sd::disc