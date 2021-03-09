#pragma once

#include <algorithm>
#include <desc/Support.hxx>
#include <desc/distribution/InferProbabilities.hxx>
#include <desc/distribution/StaticFactorModel.hxx>
#include <desc/utilities/FactorPruning.hxx>

// #include <relent/ReaperModelPruning.hxx>

namespace sd::disc
{

template <typename Dist>
void prune_distribution(Dist& pr)
{
    // auto& phi = pr.factors;
    // std::for_each(std::execution::seq_par, phi.begin(), phi.end(), [](auto& p) {
    //     if (p.factor.itemset.set.size() > 2)
    //         viva::prune_factor(p, pr.max_factor_size);
    // });

#pragma omp parallel for if (pr.factors.size() > 16)
    for (size_t i = 0; i < pr.factors.size(); ++i)
    {
        if (pr.factors[i].factor.itemsets.set.size() > 2)
        {
            viva::prune_factor(pr.factors[i], pr.max_factor_size);
        }
    }
}

template <typename Trait>
void remove_unused_patterns(disc::Component<Trait>& c, Config const&)
{
    for (size_t j = 0; j < c.summary.size();)
    {
        bool keep = std::any_of(
            c.model.model.factors.begin(), c.model.model.factors.end(), [&](const auto& f) {
                return f.factor.get_precomputed_expectation(c.summary.point(j)).has_value();
            });
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
        auto& phi = c.models[i].model.factors;

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

        for (size_t k = 0; k < c.models.size(); ++k)
        {
            const auto& phi = c.models[k].model.factors;
            keep |= std::any_of(phi.begin(), phi.end(), [&](const auto& f) {
                return f.factor.get_precomputed_expectation(x).has_value();
            });

            if (keep) break;
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
    prune_distribution(c.model.model);
    remove_unused_patterns(c, cfg);
    compute_frequency_matrix_column(c);
}

template <typename Trait>
void prune_pattern_composition(disc::Composition<Trait>& c, Config const& cfg)

{
    for (auto& m : c.models) prune_distribution(m.model);
    remove_unused_patterns(c, cfg);
}

template <typename C, typename Config>
void prune_model(C& c, const Config& cfg)
{
    // using dist_t = typename std::decay_t<C>::distribution_type;
    // if constexpr (is_dynamic_factor_model<dist_t>())
    // {
    //     // reaper_prune_pattern_composition(c, cfg);
    // }
    // else
    {
        prune_pattern_composition(c, cfg);
    }
}

template <typename C, typename Candidate, typename Config>
bool is_allowed_wrapped(const C& c, const Candidate& x, const Config& cfg)
{
    // using dist_t = typename std::decay_t<C>::distribution_type;
    // if constexpr (is_dynamic_factor_model<dist_t>()) { return true; }
    // else
    // {
        return DefaultPatternsetMinerInterface::is_allowed(c, x, cfg);
    // }
}

} // namespace sd::disc