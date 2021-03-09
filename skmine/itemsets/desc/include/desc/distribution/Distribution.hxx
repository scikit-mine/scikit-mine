#pragma once

#include <desc/distribution/InferProbabilities.hxx>
#include <desc/distribution/IterativeScaling.hxx>
#include <desc/storage/Itemset.hxx>

#include <desc/Settings.hxx>

namespace sd
{
namespace disc
{

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
    void insert(float_type label, const T& t, bool estimate = false)
    {
        label = std::clamp<float_type>(label, epsilon, float_type(1.0) - epsilon);
        model.insert(label, t, estimate);
    }
    template <typename T>
    void insert_singleton(float_type label, const T& t, bool estimate = false)
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
        return viva::probability(model, t);
    }
    template <typename pattern_t>
    auto expectation(const pattern_t& t) const
    {
        return viva::expectation(model, t);
    }
    template <typename pattern_t>
    auto expectation_generalized_set(const pattern_t& t) const
    {
        return viva::expectation_generalized_set(model, t);
    }
    template <typename pattern_t>
    auto log_probability(const pattern_t& t) const
    {
        return std::log2(viva::probability(model, t));
    }
    template <typename pattern_t>
    auto log_expectation(const pattern_t& t) const
    {
        return viva::log_expectation(model, t);
    }
    template <typename pattern_t>
    auto log_expectation_generalized_set(const pattern_t& t) const
    {
        return viva::log_expectation_generalized_set(model, t);
    }
    underlying_model_type model;
    /// Laplacian Smoothing
    ///     makes sure that the support all distributions is the complete domain.
    ///     prevents both log p or log (1- p) from being -inf.
    float_type epsilon{1e-16};
};

template <typename M, typename T = typename M::float_type>
auto estimate_model(Distribution<M>& m, viva::IterativeScalingSettings<T> const& opts = {})
{
    return viva::estimate_model(m.model, opts);
}

template <typename U, typename V>
struct MaxEntDistribution : Distribution<viva::StaticFactorModel<U, V>>
{
    using base           = Distribution<viva::StaticFactorModel<U, V>>;
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

} // namespace disc
} // namespace sd