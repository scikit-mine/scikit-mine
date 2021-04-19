#pragma once

#include <desc/distribution/IterativeScaling.hxx>
#include <desc/distribution/StaticFactorModel.hxx>
#include <desc/distribution/RelaxedFactorModel.hxx>
#include <desc/storage/Itemset.hxx>
#include <desc/Settings.hxx>

namespace sd
{
namespace disc
{

template <typename Model, typename X>
auto log_expectation(Model const& m, X const& x)
{
    using std::log2;

    using float_type   = typename Model::float_type;
    using pattern_type = typename Model::pattern_type;

    thread_local itemset<pattern_type> part;

    float_type fr = 0;
    factorize(m, x, [&](const auto& f, size_t, bool s) {
        if (s) { fr += log2(f.factor.singletons.set.front().probability); }
        else
        {
            part.clear();
            intersection(x, f.range, part);
            fr += log2(expectation(f.factor, part));
        }
    });
    return fr;
}

template <typename Model, typename X>
auto log_probability(Model const& m, X const& x)
{
    using std::log2;

    using float_type   = typename Model::float_type;
    using pattern_type = typename Model::pattern_type;

    thread_local itemset<pattern_type> part;

    float_type fr = 0;
    factorize(m, x, [&](const auto& f, size_t, bool s) {
        if (s) { fr += log2(f.factor.singletons.set.front().probability); }
        else
        {
            part.clear();
            intersection(x, f.range, part);
            fr += log2(probability(f.factor, part));
        }
    });
    return fr;
}

template <typename U, typename X>
auto log_probability_of_absent_items(const std::vector<Factor<U>>& phi, const X& x)
{
    using std::log2;
    using float_type = typename U::float_type;
    float_type p     = 0;
    for (size_t i = 0; i < phi.size(); ++i)
    {
        const auto& e = phi[i].factor.singletons.set;
        if (e.size() == 1 && !is_subset(e.front().element, x))
        {
            p += log2(1 - e.front().probability);
        }
    }
    return p;
}

template <typename U, typename X>
auto log_probability_of_absent_items(const Factorization<U>& phi, const X& x)
{
    using std::log2;
    using float_type = typename U::float_type;
    float_type p     = 0;
    for (size_t i = 0; i < phi.singleton_factors.size(); ++i)
    {
        const auto& e = phi.singleton_factors[i].factor.singletons.set;
        if (!is_subset(e.front().element, x)) { p += log2(1 - e.front().probability); }
    }
    return p;
}

// template <typename S, typename T, typename U, typename X>
// auto log_probability_of_absent_items(const RelaxedFactorModel<S, T, U>& pr, const X& x)
// {
//     return log_probability_of_absent_items(pr.phi, x);
// }
template <typename S, typename T, typename U, typename X>
auto log_probability_of_absent_items(const StaticFactorModel<S, T, U>& pr, const X& x)
{
    return log_probability_of_absent_items(pr.phi, x);
}
template <typename S, typename T, typename U, typename X>
auto log_probability_of_absent_items(const ClassicStaticFactorModel<S, T, U>& pr, const X& x)
{
    return log_probability_of_absent_items(pr.phi.factors, x);
}

template <typename Model, typename X>
auto log_expectation_generalized_set(const Model& m, const X& x)
{
    using std::log2;
    return log_expectation(m, x) + log_probability_of_absent_items(m, x);
}

template <typename Model, typename X>
auto expectation(Model const& m, X const& x)
{
    using std::exp2;
    return exp2(log_expectation(m, x));
}

template <typename Model, typename X>
auto probability(Model const& m, X const& x)
{
    using std::exp2;
    return exp2(log_probability(m, x));
}
template <typename Model, typename X>
auto expectation_generalized_set(Model const& m, X const& x)
{
    using std::exp2;
    return exp2(log_expectation_generalized_set(m, x));
}


template <typename Model>
struct Distribution
{
    using underlying_model_type = Model;
    using float_type            = typename underlying_model_type::float_type;
    using pattern_type          = typename underlying_model_type::pattern_type;

    Distribution() = default;

    Distribution(size_t dimension, size_t length)
        : model(dimension)
        , epsilon(std::min(float_type(1e-16), float_type(1) / (length + dimension)))
    {
        assert(dimension > 0);
        assert(length > 0);
    }

    void   clear() { model.clear(); }
    size_t dimension() const { return model.dimension(); }
    size_t num_itemsets() const { return model.num_itemsets(); }
    size_t size() const { return model.size(); }

    template <typename T>
    void insert(float_type label, const T& t, bool estimate)
    {
        label = std::clamp<float_type>(label, epsilon, float_type(1.0) - epsilon);
        model.insert(label, t, estimate);
    }
    template <typename T>
    void insert_singleton(float_type label, const T& t, bool estimate)
    {
        label = std::clamp<float_type>(label, epsilon, float_type(1.0) - epsilon);
        model.insert_singleton(label, t, estimate);
    }
    template <typename T>
    bool is_allowed(const T& t) const
    {
        return model.is_allowed(t);
    }
    template <typename pattern_t>
    auto probability(const pattern_t& t) const
    {
        return disc::probability(model, t);
    }
    template <typename pattern_t>
    auto expectation(const pattern_t& t) const
    {
        return disc::expectation(model, t);
    }
    template <typename pattern_t>
    auto expectation_generalized_set(const pattern_t& t) const
    {
        return disc::expectation_generalized_set(model, t);
    }
    template <typename pattern_t>
    auto log_probability(const pattern_t& t) const
    {
        return std::log2(disc::probability(model, t));
    }
    template <typename pattern_t>
    auto log_expectation(const pattern_t& t) const
    {
        return disc::log_expectation(model, t);
    }
    template <typename pattern_t>
    auto log_expectation_generalized_set(const pattern_t& t) const
    {
        return disc::log_expectation_generalized_set(model, t);
    }
    underlying_model_type model;
    /// Laplacian Smoothing
    ///     makes sure that the support all distributions is the complete domain.
    ///     prevents both log p or log (1- p) from being -inf.
    float_type epsilon{1e-16};
};

// template <typename M, typename T = typename M::float_type>
// auto estimate_model(Distribution<M>& m, IterativeScalingSettings<T> const& opts = {})
// {
//     return estimate_model(m.model, opts);
// }

template <typename U, typename V>
struct MaxEntDistribution : Distribution<StaticFactorModel<U, V>>
{
    using base           = Distribution<StaticFactorModel<U, V>>;
    MaxEntDistribution() = default;

    MaxEntDistribution(size_t dimension,
                       size_t length,
                       size_t max_factor_size  = 5,
                       size_t max_factor_width = 8)
        : base(dimension, length)
    {
        this->model.max_factor_size  = max_factor_size;
        this->model.max_factor_width = max_factor_width;
    }

    MaxEntDistribution(size_t dimension, size_t length, const disc::Config& cfg)
        : MaxEntDistribution(dimension, length, cfg.max_factor_size, cfg.max_factor_width)
    {
    }
};

// template <typename U, typename V>
// struct RelEntDistribution : Distribution<RelaxedFactorModel<U, V>>
// {
//     using base           = Distribution<RelaxedFactorModel<U, V>>;
//     RelEntDistribution() = default;

//     RelEntDistribution(size_t dimension,
//                        size_t length,
//                        size_t max_factor_size  = 5,
//                        size_t max_factor_width = 8)
//         : base(dimension, length)
//     {
//         this->model.max_factor_size  = max_factor_size;
//         this->model.max_factor_width = max_factor_width;
//     }

//     RelEntDistribution(size_t dimension, size_t length, const disc::Config& cfg)
//         : RelEntDistribution(dimension, length, cfg.max_factor_size, cfg.max_factor_width)
//     {
//     }
// };



template <typename D>
constexpr bool is_static_factor_model()
{
    using U  = std::decay_t<D>;
    using S  = typename U::pattern_type;
    using T  = typename U::float_type;
    using N0 = StaticFactorModel<S, T>;
    using N1 = Distribution<StaticFactorModel<S, T>>;
    using N2 = MaxEntDistribution<S, T>;
    using N3 = ClassicStaticFactorModel<S, T>;
    using N4 = Distribution<ClassicStaticFactorModel<S, T>>;

    return std::is_same_v<U, N0> || std::is_same_v<U, N1> || std::is_same_v<U, N2> ||
           std::is_same_v<U, N3> || std::is_same_v<U, N4>;
}

template <typename D>
constexpr bool is_dynamic_factor_model()
{
    return !is_static_factor_model<D>();
}

} // namespace disc
} // namespace sd