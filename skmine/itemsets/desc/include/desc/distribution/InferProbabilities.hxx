#pragma once

#include <desc/distribution/MaxEntFactor.hxx>
#include <desc/distribution/StaticFactorModel.hxx>
#include <desc/distribution/Transactions.hxx>
#include <desc/storage/Itemset.hxx>

#include <numeric>

namespace sd
{
namespace viva
{

template <typename pattern_type, typename float_type, typename query_type>
auto probability(ItemsetModel<pattern_type, float_type> const& c, const query_type& t)
{
    float_type acc = c.theta0;
    for (size_t i = 0, l = c.set.size(); i < l; ++i)
    {
        if (is_subset(c.set[i].point, t))
        {
            acc *= c.set[i].theta;
        }
    }
    return acc;
}

template <typename pattern_type, typename float_type, typename query_type>
auto probability(SingletonModel<pattern_type, float_type> const& c, query_type const& t)
{
    float_type acc = c.theta0;
    for (size_t i = 0, l = c.set.size(); i < l; ++i)
    {
        if (is_subset(c.set[i].element, t))
        {
            acc *= c.set[i].theta;
        }
    }
    return acc;
}

template <typename pattern_type, typename float_type, typename query_type>
auto probability(MaxEntFactor<pattern_type, float_type> const& c, query_type const& t)
{
    return probability(c.itemsets, t) * probability(c.singletons, t);
}

template <typename pattern_type,
          typename float_type,
          typename underlying_factor,
          typename query_type>
auto probability(StaticFactorModel<pattern_type, float_type, underlying_factor> const& c,
                 query_type const&                                                     t)
{
    return std::accumulate(
        c.factors.begin(), c.factors.end(), float_type(1), [&t](auto acc, const auto& f) {
            return acc * probability(f.factor, t);
        });
}

template <typename Transactions, typename Model, typename Pattern>
auto expectation_known(Transactions const& transactions,
                       size_t              len,
                       Model const&        model,
                       Pattern const&      x)
{
    using float_type = typename Model::float_type;

    float_type p = 0;

    for (size_t i = 0; i < len; ++i)
    {
        const auto& t = transactions[i];
        if (t.value != 0 && is_subset(x, t.cover))
        {
            p += t.value * probability(model, t.cover);
            assert(!std::isnan(p));
            assert(!std::isinf(p));
        }
    }

    return p;
}

template <typename Transactions, typename Model, typename Pattern>
auto expectation_known(Transactions const& transactions, Model const& model, Pattern const& x)
{
    return expectation_known(transactions, transactions.size(), model, x);
}

template <typename Model_Type>
size_t dimension_of_factor(const Model_Type& m)
{
    return m.singletons.size();
}
template <typename Model_Type, typename T>
size_t dimension_of_factor(const Model_Type& m, const T&)
{
    return dimension_of_factor(m);
}

template <typename Pattern_Type, typename Float_type>
struct AugmentedModel
{
    using pattern_type = Pattern_Type;
    using float_type   = Float_type;

    AugmentedModel(ItemsetModel<pattern_type, float_type> const& underlying_model,
                   disc::itemset<pattern_type> const&            additional_pattern)
        : underlying_model(underlying_model), additional_pattern(additional_pattern)
    {
    }

    size_t size() const { return underlying_model.size() + 1; }

    const auto& point(size_t i) const
    {
        if (i < underlying_model.size())
        {
            return underlying_model.point(i);
        }
        else
        {
            return additional_pattern;
        }
    }

    ItemsetModel<pattern_type, float_type> const& underlying_model;
    disc::itemset<pattern_type> const&            additional_pattern;
};

template <typename S, typename T>
auto augment_model(MaxEntFactor<S, T> const& model, disc::itemset<S> const& x)
{
    return AugmentedModel<S, T>(model.itemsets, x);
}

template <typename Blocks, typename S, typename T>
auto make_partitions_for_unknown(Blocks&                   b,
                                 MaxEntFactor<S, T> const& model,
                                 disc::itemset<S> const&   x)
{
    return viva::compute_counts(dimension_of_factor(model, x), augment_model(model, x), b);
}

template <typename S, typename T, size_t N = 13>
struct Temp_Partition_Buffer
{
    Temp_Partition_Buffer()
    {
        for (size_t i = 0; i < N; ++i)
        {
            blocks_of_size[i].resize(1 << i);
        }
    }

    auto& get(size_t i)
    {
        if (i < N)
            return blocks_of_size[i];
        else
            return rest;
    }

    std::array<std::vector<Block<S, T>>, N> blocks_of_size;
    std::vector<Block<S, T>>                rest;
};

template <typename S, typename T>
auto expectation_unknown(MaxEntFactor<S, T> const& model, disc::itemset<S> const& x)
{
    thread_local Temp_Partition_Buffer<S, T, 13> bf;

    auto& b   = bf.get(model.itemsets.set.size() + 1);
    auto  len = make_partitions_for_unknown(b, model, x);

    return expectation_known(b, len, model, x);
}

template <typename S, typename T, typename Blocks>
auto compute_transactions(MaxEntFactor<S, T> const& model,
                          disc::itemset<S> const&   x,
                          bool                      known,
                          Blocks&                   blocks)
{
    if (!known)
    {
        return viva::compute_counts(
            dimension_of_factor(model, x), augment_model(model, x), blocks);
    }
    else
    {
        return viva::compute_counts(dimension_of_factor(model, x), model.itemsets, blocks);
    }
}

template <typename S, typename T>
auto expectation(MaxEntFactor<S, T> const& model, disc::itemset<S> const& x)
{
    if (auto p = model.get_precomputed_expectation(x); p) // && !is_singleton(x))
    {
        // return expectation_known(model.itemsets.partitions, model, x) ;
        return p.value();
    }
    else
    {
        return expectation_unknown(model, x);
    }
}

template <typename Model, typename query_type>
auto probability_of_absent_items(Model const& m, query_type const& t)
{
    using float_type = typename Model::float_type;
    float_type p     = 1;
    for (const auto& s : m.singletons.set)
    {
        if (!is_subset(s.element, t))
        {
            p *= float_type(1.0) - s.probability;
        }
    }
    return p;
}

template <typename Model, typename query_type>
auto log_probability_of_absent_items(Model const& m, query_type const& t)
{
    using float_type = typename Model::float_type;
    float_type p     = 0;
    for (const auto& s : m.singletons.set)
    {
        if (!is_subset(s.element, t))
        {
            p += std::log2(float_type(1.0) - s.probability);
        }
    }
    return p;
}

template <typename Model, typename query_type>
auto expectation_generalized_set(Model const& m, query_type const& t)
{
    return expectation(m, t) * probability_of_absent_items(m, t);
}

template <typename pattern_type, typename float_type, typename Underlying, typename query_type>
auto expectation(const StaticFactorModel<pattern_type, float_type, Underlying>& fm,
                 const query_type&                                              t)
{
    thread_local disc::itemset<pattern_type> part;
    part.clear();
    part.reserve(fm.dim);

    float_type estimate = 1;

    for (size_t i = 0; i < fm.factors.size(); ++i)
    {
        const auto& f = fm.factors[i];
        if (intersects(t, f.range))
        {
            intersection(t, f.range, part);
            estimate *= expectation(f.factor, part);
        }
    }
    return estimate;
}

template <typename pattern_type, typename float_type, typename Underlying, typename query_type>
auto log_expectation(const StaticFactorModel<pattern_type, float_type, Underlying>& fm,
                     const query_type&                                              t)
{
    thread_local disc::itemset<pattern_type> part;
    part.clear();
    part.reserve(fm.dim);

    auto smooth_if = [lb = float_type(1) / (fm.dim * fm.dim)](auto p) {
        if (p <= 0 || p > 1)
        {
            return std::clamp(p, lb, float_type(1) - lb);
        }
        return p;
    };

    float_type estimate = 0;
    for (size_t i = 0; i < fm.factors.size(); ++i)
    {
        const auto& f = fm.factors[i];
        if (intersects(t, f.range))
        {
            intersection(t, f.range, part);
            estimate += std::log2(smooth_if(expectation(f.factor, part)));
        }
    }
    return estimate;
}

template <typename query_type, typename pattern_type, typename float_type, typename Underlying>
auto expectation_generalized_set(
    const StaticFactorModel<pattern_type, float_type, Underlying>& m, const query_type& t)
{
    float_type p = expectation(m, t);
    for (const auto& f : m.factors)
    {
        p *= probability_of_absent_items(f.factor, t);
    }
    return p;
}

template <typename query_type, typename pattern_type, typename float_type, typename Underlying>
auto log_expectation_generalized_set(
    const StaticFactorModel<pattern_type, float_type, Underlying>& m, const query_type& t)
{
    float_type p = log_expectation(m, t);
    for (const auto& f : m.factors)
    {
        p += log_probability_of_absent_items(f.factor, t);
    }
    return p;
}

} // namespace viva
} // namespace sd
