#pragma once

#include <desc/Composition.hxx>
#include <desc/PatternAssignment.hxx>
#include <desc/Support.hxx>

#include <limits>

namespace sd
{
namespace disc
{

template <typename Trait, typename Interface = DefaultAssignment>
void characterize_one_component(Composition<Trait>& c,
                                size_t              index,
                                const Config&       cfg,
                                Interface&&         f = {})
{
    c.models[index] = make_distribution(c, cfg);
    c.assignment[index].clear();

    using float_t = typename Trait::float_type;

    for (size_t i = 0; i < c.summary.size(); ++i)
    {
        const auto& x = c.summary.point(i);
        if (is_singleton(x))
        {
            c.assignment[index].insert(i);
            c.confidence(i, index) = std::numeric_limits<float_t>::infinity();
            c.models[index].insert_singleton(c.frequency(i, index), x, true);
        }
    }

    // separate stages: depends on singletons.
    for (size_t i = 0; i < c.summary.size(); ++i)
    {
        const auto& x = c.summary.point(i);
        const auto& q = c.frequency(i, index);

        if (is_singleton(x)) continue;

        c.confidence(i, index) =
            c.models[index].is_allowed(x) ? f.confidence(c, index, q, x, cfg) : float_t(0);

        if (c.confidence(i, index))
        {
            c.assignment[index].insert(i);
            c.models[index].insert(q, x, true);
        }
    }
}

template <typename Trait, typename Interface = DefaultAssignment>
void characterize_no_mining(Composition<Trait>& c, const Config& cfg, Interface&& f = {})
{
    compute_frequency_matrix(c);

    c.confidence.resize(sd::layout<2>({c.summary.size(), c.data.num_components()}), 0);
    c.assignment.assign(c.data.num_components(), {});
    c.models.assign(c.data.num_components(), make_distribution(c, cfg));

    for (size_t j = 0; j < c.data.num_components(); ++j)
    {
        characterize_one_component(c, j, cfg, f);
    }
}

template <typename Trait, typename Interface = DefaultAssignment>
void characterize_components(Composition<Trait>& c, const Config& cfg, Interface&& f = {})
{
    characterize_no_mining(c, cfg, std::forward<Interface>(f));
}

template <typename Trait, typename Interface = DefaultAssignment>
void characterize_components(Component<Trait>& c, const Config& cfg = {}, Interface&& f = {})
{
    compute_frequency_matrix_column(c);
    c.confidence.assign(c.summary.size(), 0);
    c.model = make_distribution(c, cfg);

    using float_t = typename Trait::float_type;

    for (size_t i = 0; i < c.summary.size(); ++i)
    {
        const auto& x = c.summary.point(i);
        if (is_singleton(x))
        {
            c.confidence[i] = std::numeric_limits<float_t>::infinity();
            c.model.insert_singleton(c.frequency[i], x, true);
        }
    }

    // separate stages: depends on singletons.
    for (size_t i = 0; i < c.summary.size(); ++i)
    {
        const auto& x = c.summary.point(i);
        const auto& q = c.frequency[i];

        if (is_singleton(x)) continue;

        c.confidence[i] = c.model.is_allowed(x) ? f.confidence(c, q, x, cfg) : float_t(0);

        if (c.confidence[0]) { c.model.insert(q, x, true); }
    }
}

struct TrueAssignment
{
    template <class... T>
    bool confidence(T&&...) const
    {
        return true;
    }
};

template <typename Trait, typename Interface = TrueAssignment>
void initialize_model(Component<Trait>& c, const Config& cfg = {}, Interface&& f = {})
{
    c.data.dim    = std::max(c.data.dim, c.summary.dim);
    c.summary.dim = c.data.dim;
    insert_missing_singletons(c.data, c.summary);
    characterize_components(c, cfg, std::forward<Interface>(f));
}

template <typename Trait, typename Interface = DefaultAssignment>
void initialize_model(Composition<Trait>& c, const Config& cfg = {}, Interface&& f = {})
{
    c.data.group_by_label();
    insert_missing_singletons(c.data, c.summary);
    characterize_no_mining(c, cfg, std::forward<Interface>(f));
    c.masks = construct_component_masks(c);
    assert(check_invariant(c));
}

} // namespace disc
} // namespace sd
