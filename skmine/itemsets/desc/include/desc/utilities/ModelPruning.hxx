#pragma once

#include <bitcontainer/sparse_bitset.hxx>
#include <desc/Support.hxx>
#include <desc/distribution/StaticFactorModel.hxx>
#include <desc/storage/Itemset.hxx>
#include <desc/utilities/FactorPruning.hxx>
#include <ndarray/ndarray.hxx>

#include <algorithm>
#include <vector>

namespace sd::disc
{

template <typename Factors>
void clear_unused_factors(const std::vector<size_t>& u, Factors& facs)
{
    for (size_t i = 0; i < u.size(); ++i)
    {
        if (u[i] == 0 && !is_singleton(facs[i].range))
        {
            facs[i].factor.itemsets.set.clear();
            facs[i].factor.singletons.set.clear();
            facs[i].range.clear();
        }
    }
}

template <typename U, typename Data>
void calc_factor_usage(std::vector<size_t>& u, U const& m, const Data& data)
{
    u.assign(m.phi.factors.size(), 0);

    for (const auto& x : data)
    {
        factorize_patterns(m, point(x), [&](auto const&, size_t index) { u[index]++; });
    }
}

template <typename Factors>
void remove_unused_factors(const std::vector<size_t>& u, Factors& facs)
{
    clear_unused_factors(u, facs);
    facs.erase(
        std::remove_if(facs.begin(), facs.end(), [](const auto& f) { return f.range.empty(); }),
        facs.end());
}

template <typename Trait>
void prune_unused_factors(Component<Trait>& c)
{
    std::vector<size_t> u;
    calc_factor_usage(u, c.model.model, c.data);
    remove_unused_factors(u, c.model.model.phi.factors);
}

template <typename Trait>
void prune_unused_factors(Composition<Trait>& c)
{
    std::vector<size_t> u;
    for (auto& m : c.models)
    {
        calc_factor_usage(u, m.model, c.data);
        remove_unused_factors(u, m.model.phi.factors);
    }
}

template <typename U>
void prune_individual_factors(std::vector<Factor<U>>& phi, size_t max_factor_size)
{
#if defined(HAS_EXECUTION_POLICIES)
    std::for_each(std::execution::par_unseq, phi.begin(), phi.end(), [&](auto& phi_i) {
        if (phi_i.factor.itemsets.set.size() > 2) prune_factor(phi_i, max_factor_size);
    });
#else
#pragma omp parallel for if (phi.size() > 16)
    for (size_t i = 0; i < phi.size(); ++i)
    {
        if (phi[i].factor.itemsets.set.size() > 2) { prune_factor(phi[i], max_factor_size); }
    }
#endif
}

template <typename Trait>
void prune_individual_factors(Component<Trait>& c, size_t max_factor_size)
{
    prune_individual_factors(c.model.model.phi.factors, max_factor_size);
}
template <typename Trait>
void prune_individual_factors(Composition<Trait>& c, size_t max_factor_size)
{
    for (auto& m : c.models) prune_individual_factors(m.model.phi.factors, max_factor_size);
}

template <class T>
void erase_row(sd::ndarray<T, 2>& x, size_t row)
{
    for (size_t k = row + 1; k < x.extent(0); ++k)
        for (size_t l = 0; l < x.extent(1); ++l) x(k - 1, l) = x(k, l);
}

template <typename Trait>
void prune_unused_patterns(Composition<Trait>& c)
{
    itemset<tag_dense> cataloge;
    cataloge.reserve(c.summary.size());

    for (size_t i = 0; i < c.summary.size(); ++i)
    {
        if (is_singleton(c.summary.point(i))) { continue; }
        else if (!any_factor_contains(c, c.summary.point(i)))
        {
            cataloge.insert(i);
        }
    }

    for (size_t i = 0, m = 0; i < c.summary.size(); ++i)
    {
        if (cataloge.contains(i))
        {
            c.summary.erase_row(m);
            erase_row(c.frequency, m);
            erase_row(c.confidence, m);
        }
        else
        {
            m++;
        }
    }
}

template <typename Trait>
void prune_unused_patterns(Component<Trait>& c)
{
    itemset<tag_dense> cataloge;
    cataloge.reserve(c.summary.size());

    for (size_t i = 0; i < c.summary.size(); ++i)
    {
        if (is_singleton(c.summary.point(i))) { continue; }
        else if (!any_factor_contains(c, c.summary.point(i)))
        {
            cataloge.insert(i);
        }
    }

    for (size_t i = 0, m = 0; i < c.summary.size(); ++i)
    {
        if (cataloge.contains(i))
        {
            c.summary.erase_row(m);
            c.frequency.erase(c.frequency.begin() + m);
            c.confidence.erase(c.confidence.begin() + m);
        }
        else
        {
            m++;
        }
    }
}

template <typename Trait, typename X>
bool any_factor_contains(Component<Trait>& c, const X& x)
{
    const auto& phi = c.model.model.phi.factors;
    if (std::any_of(phi.begin(), phi.end(), [&](const auto& f) {
            return f.factor.get_precomputed_expectation(x).has_value();
        }))
    {
        return true;
    }
    else
    {
        return false;
    }
}

template <typename Trait, typename X>
bool any_factor_contains(Composition<Trait> const& c, const X& x)
{
    for (size_t k = 0; k < c.models.size(); ++k)
    {
        const auto& phi = c.models[k].model.phi.factors;
        if (std::any_of(phi.begin(), phi.end(), [&](const auto& f) {
                return f.factor.get_precomputed_expectation(x).has_value();
            }))
        {
            return true;
        }
    }
    return false;
}

template <typename Trait>
void assign_from_factors(Component<Trait> const&)
{
}

template <typename Trait>
void assign_from_factors(Composition<Trait>& c)
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

template <typename C, typename Confidence>
void prune_composition_mk1(C& c, const Config& cfg, Confidence&& f)
{
    using dist_t = typename std::decay_t<C>::distribution_type;
    if constexpr (is_dynamic_factor_model<dist_t>())
    {
        prune_unused_factors(c);
        prune_individual_factors(c, cfg.max_factor_size);
        prune_unused_patterns(c);
        assign_from_factors(c);
    }
    else
    {
        size_t before = c.summary.size();
        prune_individual_factors(c, cfg.max_factor_size);
        prune_unused_patterns(c);
        characterize_components(c, cfg, f);
        size_t after = c.summary.size();
        if (before != after) { prune_unused_patterns(c); }
    }
}

template <typename C, typename Confidence>
void prune_composition(C& c, const Config& cfg, Confidence&& f)
{
    using dist_t = typename std::decay_t<C>::distribution_type;
    if constexpr (is_dynamic_factor_model<dist_t>()) { prune_unused_factors(c); }

    size_t before = c.summary.size();
    prune_individual_factors(c, cfg.max_factor_size);
    prune_unused_patterns(c);
    characterize_components(c, cfg, f);
    size_t after = c.summary.size();
    if (before != after) { prune_unused_patterns(c); }
}

} // namespace sd::disc