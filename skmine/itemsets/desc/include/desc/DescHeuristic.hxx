#pragma once

#include <desc/Composition.hxx>
#include <desc/Component.hxx>

namespace sd::disc
{

template <typename Trait, typename Candidate>
auto desc_heuristic_multi(const Composition<Trait>& c, const Candidate& x)
{
    using float_type = typename Trait::float_type;

    float_type acc = 0;

    for (size_t i = 0; i < c.data.num_components(); ++i)
    {
        auto n = c.data.subset(i).size();
        auto p = c.models[i].expectation(x.pattern);
        auto s = size_of_intersection(x.row_ids, c.masks[i]);
        auto q = static_cast<float_type>(s) / n;
        auto h = s == 0 ? 0 : s * std::log2(q / p);

        assert(0 <= p && p <= 1);

        acc += h - std::log2(n);
    }

    return acc;
}

template <typename C, typename Distribution, typename Candidate>
auto desc_heuristic_1(const C& c, const Distribution& pr, const Candidate& x)
{
    using float_type = typename C::float_type;

    const auto s = static_cast<float_type>(x.support);
    const auto q = s / c.data.size();
    const auto p = pr.expectation(x.pattern);
    assert(0 <= p && p <= 1);
    return s * std::log2(q / p) - std::log2(c.data.size());
}

template <typename Trait, typename Candidate>
auto desc_heuristic(const Composition<Trait>& c, const Candidate& x) ->
    typename Trait::float_type
{
    if (c.data.num_components() == 1)
    {
        return desc_heuristic_1(c, c.models.front(), x);
    }
    else
    {
        return desc_heuristic_multi(c, x);
    }
}

template <typename Trait, typename Candidate>
auto desc_heuristic(const Component<Trait>& c, const Candidate& x) -> typename Trait::float_type
{
    return desc_heuristic_1(c, c.model, x);
}

} // namespace sd::disc