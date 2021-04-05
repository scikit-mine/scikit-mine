#pragma once

#include <algorithm>
#include <desc/Support.hxx>
#include <desc/distribution/StaticFactorModel.hxx>
#include <desc/utilities/FactorPruning.hxx>
#include <vector>
// #include <desc/PatternsetMiner.hxx>

namespace sd::disc
{

template <typename U>
void prune_factorization(std::vector<Factor<U>>& phi, size_t max_factor_size)
{
#if defined(HAS_EXECUTION_POLICIES)
    std::for_each(std::execution::par_unseq, phi.begin(), phi.end(), [&](auto& phi_i) {
        if (phi_i.factor.itemsets.set.size() > 2) disc::prune_factor(phi_i, max_factor_size);
    });
#else
#pragma omp parallel for if (phi.size() > 16)
    for (size_t i = 0; i < phi.size(); ++i)
    {
        if (phi[i].factor.itemsets.set.size() > 2)
        {
            disc::prune_factor(phi[i], max_factor_size);
        }
    }
#endif
}

template <typename Trait>
void remove_unused_patterns(disc::Component<Trait>& c, Config const&)
{
    for (size_t j = 0; j < c.summary.size();)
    {
        bool keep = false;
        if (is_singleton(c.summary.point(j))) { keep = true; }
        else
        {
            keep = std::any_of(
                c.model.model.phi.factors.begin(),
                c.model.model.phi.factors.end(),
                [&](const auto& f) {
                    return f.factor.get_precomputed_expectation(c.summary.point(j)).has_value();
                });
        }
        if (!keep) { c.summary.erase_row(j); }
        else
        {
            ++j;
        }
    }
}

template <typename Trait>
void assign_from_factors(disc::Composition<Trait>& c)
{
    const auto& q = c.frequency;
    c.assignment.resize(c.data.num_components());

    for (size_t i = 0; i < c.assignment.size(); ++i)
    {
        auto& a   = c.assignment[i];
        auto& phi = c.models[i].model.phi.factors;

        a.clear();

        for (size_t j = 0; j < c.summary.size(); j++)
        {
            const auto& x = c.summary.point(j);

            bool keep = is_singleton(x) ||
                        (q(j, i) > 0 && std::any_of(phi.begin(), phi.end(), [&](const auto& f) {
                             return f.factor.get_precomputed_expectation(x).has_value();
                         }));
            if (keep) { a.insert(j); }
        }
    }
}

template <typename Trait>
void remove_unused_patterns_from_summary(disc::Composition<Trait>& c)
{
    for (size_t j = 0; j < c.summary.size();)
    {
        const auto& x    = c.summary.point(j);
        bool        keep = false;
        
        if(is_singleton(c.summary.point(j)))
            keep = true;

        for (size_t k = 0; k < c.models.size() && !keep; ++k)
        {
            const auto& phi = c.models[k].model.phi.factors;
            keep |= std::any_of(phi.begin(), phi.end(), [&](const auto& f) {
                return f.factor.get_precomputed_expectation(x).has_value();
            });
        }

        if (!keep) { c.summary.erase_row(j); }
        else
        {
            ++j;
        }
    }
}

template <typename Trait>
void remove_unused_patterns(disc::Composition<Trait>& c, Config const&)
{
    remove_unused_patterns_from_summary(c);
    // characterize_components(c, cfg);
    assign_from_factors(c);
    compute_frequency_matrix(c);
}

template <typename Trait>
void prune_pattern_composition(disc::Component<Trait>& c, Config const& cfg)
{
    prune_factorization(c.model.model.phi.factors, cfg.max_factor_size);
    remove_unused_patterns(c, cfg);
    compute_frequency_matrix_column(c);
}

template <typename Trait>
void prune_pattern_composition(disc::Composition<Trait>& c, Config const& cfg)

{
    for (auto& m : c.models) prune_factorization(m.model.phi.factors, cfg.max_factor_size);
    remove_unused_patterns(c, cfg);
}

// template <typename C, typename Candidate, typename Config>
// bool is_allowed_wrapped(const C& c, const Candidate& x, const Config& cfg)
// {
//     using dist_t = typename std::decay_t<C>::distribution_type;
//     if constexpr (is_dynamic_factor_model<dist_t>()) { return true; }
//     else
//     {
//         return DefaultPatternsetMinerInterface::is_allowed(c, x, cfg);
//     }
// }

} // namespace sd::disc