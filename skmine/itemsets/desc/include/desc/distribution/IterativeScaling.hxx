#pragma once

#include <desc/distribution/MaxEntFactor.hxx>

#include <cmath>
#include <vector>


namespace sd
{
namespace disc
{

template <typename float_type>
struct IterativeScalingSettings
{
    float_type sensitivity   = 1e-8;
    float_type epsilon       = 1e-10;
    size_t     max_iteration = 100;
    bool       warmstart     = false;
    // bool       normalize     = false;
};

template <typename T>
bool bad_scaling_factor(const T& a, const T& z)
{
    const auto r = z * a;
    return std::isinf(r) || std::isnan(r) || r <= 0 || (r != r);
}

template <typename T>
bool bad_scaling_factor(const T& p, const T& q, const T& z)
{
    const auto r = z * (q / p);
    return std::isinf(r) || std::isnan(r) || r <= 0 || (r != r);
}

template <typename model_type, typename Transactions>
void update_precomputed_probabilities(model_type& model, [[maybe_unused]] const Transactions& t)
{
    for (size_t i = 0; i < model.size(); ++i)
    {
        model.probability(i) = expectation_known(t[i], model, model.point(i));
    }
}

template <typename U, typename V>
void reset_normalizer(MaxEntFactor<U, V>& model)
{
    model.singletons.theta0 = 1;
    model.itemsets.theta0   = std::exp2(-V(dimension_of_factor(model)));
}

template <typename U, typename V>
void reset_coefficients(MaxEntFactor<U, V>& model)
{
    for (auto& x : model.itemsets.set)
        x.theta = x.frequency;
    for (auto& x : model.singletons.set)
        x.theta = x.frequency;
    reset_normalizer(model);
}

template <typename Model, typename Transactions, typename AllTransactions, typename F>
auto iterative_scaling(Model&                           model,
                       const std::vector<Transactions>& transactions,
                       const AllTransactions&,
                       IterativeScalingSettings<F> opts)
{
    using float_type = typename Model::float_type;

    if (!opts.warmstart)
    {
        reset_coefficients(model);
    }
    else
    {
        reset_normalizer(model);
    }

    float_type pg = std::numeric_limits<float_type>::max();

    for (size_t it = 0; it < opts.max_iteration; ++it)
    {
        float_type g = 0;

        for (size_t i = 0; i < model.size(); ++i)
        {
            auto q               = model.frequency(i);
            auto p               = expectation_known(transactions[i], model, model.point(i));
            model.probability(i) = p;

            g += std::abs(q - p);

            if (std::abs(q - p) < opts.sensitivity)
                continue;
            if (bad_scaling_factor(p, q, model.coefficient(i)))
                continue;
            // if (bad_condition_number(p, q, model.normalizer())) continue;

            model.coefficient(i) *= q / p; // * ((1 - p) / (1 - q));
            // model.normalizer() *= (1 - q) / (1 - p);
        }

        if (g / model.size() < opts.sensitivity || std::abs(g - pg) < opts.epsilon)
        {
            pg = g;
            break;
        }

        pg = g;
    }
    return pg;
}

template <typename S, typename T, typename U = double>
auto estimate_model(MaxEntFactor<S, T>& m, IterativeScalingSettings<U> const& opts = {})
{
    using float_type   = typename MaxEntFactor<S, T>::float_type;
    using pattern_type = typename MaxEntFactor<S, T>::pattern_type;
    using block_t      = Block<pattern_type, float_type>;

    assert(m.size() > 0);

    thread_local std::vector<std::vector<block_t>> t;
    thread_local std::vector<block_t>              partitions;

    m.itemsets.num_singletons = m.singletons.set.size();

    const bool use_one_set = false && m.singletons.set.size() < m.itemsets.set.size();
    if (use_one_set)
    {
        auto n = compute_counts(m.itemsets.num_singletons, m.singletons, partitions);
        partitions.resize(n);
    }
    else
    {
        partitions.clear();
    }

    t.resize(m.size());

    // #pragma omp parallel for shared(t) if (m.size() > 16)
    for (size_t i = 0; i < m.size(); ++i)
    {
        if (m.is_pattern_known(i))
        {
            if (use_one_set && !partitions.empty())
            {
                t[i] = partitions;
            }
            else
            {
                auto n = compute_transactions(m, m.point(i), true, t[i]);
                t[i].resize(n);
            }
        }
        else
        {
            auto n = compute_transactions(m, m.point(i), false, t[i]);
            t[i].resize(n);
        }

        auto it = std::remove_if(t[i].begin(), t[i].end(), [&](const auto& x) {
            return x.value == 0 || !is_subset(m.point(i), x.cover);
        });
        t[i].erase(it, t[i].end());
    }

    const auto g = iterative_scaling(m, t, partitions, opts);

    update_precomputed_probabilities(m, t);

    return g;
}

} // namespace disc
} // namespace sd
