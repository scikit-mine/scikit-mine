#pragma once

#include <desc/distribution/IterativeScaling.hxx>
#include <desc/distribution/MaxEntFactor.hxx>
#include <desc/storage/Itemset.hxx>

#ifndef NDEBUG
#include <exception>
#endif

#if defined(HAS_EXECUTION_POLICIES)
#include <algorithm>
#include <execution>
#endif

namespace sd
{
namespace disc
{

template <typename Underlying_Factor_Type>
struct Factor
{
    using pattern_type = typename Underlying_Factor_Type::pattern_type;
    explicit Factor(size_t dim) : factor(dim) { range.reserve(dim); }
    itemset<pattern_type>  range;
    Underlying_Factor_Type factor;
};

template <typename U>
void join_factors(Factor<U>& f, const Factor<U>& g)
{
    for (auto& t : g.factor.itemsets.set) f.factor.insert_pattern(t.frequency, t.point, false);

    for (auto& t : g.factor.singletons.set) f.factor.insert_singleton(t.frequency, t.element, false);

    f.range.insert(g.range);
}

template <typename U>
struct Factorization
{
    using float_type = typename U::float_type;

    std::vector<Factor<U>> singleton_factors;
    std::vector<Factor<U>> factors;
};

template <typename U>
void init_singletons(std::vector<Factor<U>>& factors, size_t dim)
{
    factors.clear();
    factors.clear();
    factors.resize(dim, Factor<U>(dim));

    for (size_t i = 0; i < dim; ++i)
    {
        factors[i].range.insert(i);
        factors[i].factor.insert_singleton(0.5, i, true);
    }
}

template <typename U, typename float_type>
void set_singleton(std::vector<Factor<U>>& factors,
                   float_type              frequency,
                   const size_t            element,
                   bool                    estimate)
{
    factors[element].factor.insert_singleton(frequency, element, estimate);
    factors[element].range.insert(element);
}

template <typename U, typename float_type>
void insert_singleton(std::vector<Factor<U>>& factors,
                      size_t                  dim,
                      float_type              frequency,
                      const size_t            element,
                      bool                    estimate)
{
    bool add_new = true;
    for (auto& f : factors)
    {
        if (is_subset(element, f.range))
        {
            f.factor.insert_singleton(frequency, element, estimate);
            if (is_singleton(f.range)) { add_new = false; }
        }
    }
    if (add_new)
    {
        auto& f = factors.emplace_back(dim);
        f.range.insert(element);
        f.factor.insert_singleton(frequency, element, estimate);
    }
}

template <typename U, typename F = double>
void estimate_model(std::vector<U>& phi, IterativeScalingSettings<F> const& opts = {})
{
#if HAS_EXECUTION_POLICIES
    std::for_each(std::execution::par_unseq, phi.begin(), phi.end(), [opts](auto& f) {
        estimate_model(f.factor, opts);
    });
#else
#pragma omp parallel for
    for (size_t i = 0; i < phi.size(); ++i) { estimate_model(phi[i].factor, opts); }
#endif
}

template <typename U>
void erase_empty_factors(Factorization<U>& phi)
{
    phi.factors.erase(std::remove_if(phi.factors.begin(),
                                     phi.factors.end(),
                                     [](auto& f) { return f.range.empty(); }),
                      phi.factors.end());
    // phi.singleton_factors.erase(
    //     std::remove_if(phi.singleton_factors.begin(), phi.singleton_factors.end(), [](auto&
    //     f) { return f.range.empty(); }), phi.singleton_factors.end());
}

template <typename U, typename X, typename Visitor>
void factorize_classic(std::vector<Factor<U>> const& phi, const X& x, Visitor&& f)
{
    for (size_t i = 0; i < phi.size(); ++i)
    {
        if (intersects(x, phi[i].range)) { f(phi[i], i); }
    }
}

template <typename U, typename X, typename Visitor>
void factorize_patterns_statically(const std::vector<Factor<U>>& factors, X& x, Visitor&& f)
{
    for (size_t i = 0; i < factors.size(); ++i)
    {
        if (intersects(x, factors[i].range))
        {
            f(factors[i], i);
            setminus(x, factors[i].range);
        }
    }
}

template <typename Underlying_Factor_Type, typename X, typename Visitor>
void factorize_statically(const Factorization<Underlying_Factor_Type>& phi,
                          const X&                                     x,
                          Visitor&&                                    f)
{
    using pattern_type = typename Underlying_Factor_Type::pattern_type;

    thread_local itemset<pattern_type> y;
    y.clear();
    y.assign(x);

    factorize_patterns_statically(
        phi.factors, y, [&](const auto& phi_i, size_t i) { f(phi_i, i, false); });
    foreach (y, [&](size_t i) { f(phi.singleton_factors[i], i, true); })
        ;
}

template <typename U, typename V, typename Underlying_Factor_Type = MaxEntFactor<U, V>>
struct StaticFactorModel
{
    using float_type            = V;
    using pattern_type          = U;
    using index_type            = std::size_t;
    using underlying_model_type = Underlying_Factor_Type;
    using factor_type           = Factor<underlying_model_type>;

    constexpr static bool enable_factor_pruning = false;

    explicit StaticFactorModel(size_t dim) : dim(dim) { init(dim); }
    StaticFactorModel() = default;

    size_t dim              = 0;
    size_t max_factor_size  = 5;
    size_t max_factor_width = 8;

    Factorization<Underlying_Factor_Type> phi;

    size_t num_itemsets() const { return phi.factors.size(); }
    size_t dimension() const { return dim; }
    size_t size() const { return num_itemsets() + dimension(); }

    void init(size_t d)
    {
        dim = d;
        clear();
        init_singletons(phi.singleton_factors, dim);
    }

    void clear()
    {
        phi.factors.clear();
        phi.singleton_factors.clear();
    }

    void insert_singleton(float_type frequency, const index_type element, bool estimate)
    {
        set_singleton(phi.singleton_factors, frequency, element, estimate);
        // disc::insert_singleton(phi.singleton_factors, dim, frequency, element, estimate);
    }

    template <typename T>
    void insert_singleton(float_type frequency, const T& t, bool estimate)
    {
        insert_singleton(frequency, static_cast<index_type>(front(t)), estimate);
    }

    void prune_factor_if(factor_type& f, size_t max_factor_size)
    {
        if constexpr (enable_factor_pruning)
        {
            if (f.factor.itemsets.set.size() > 3) { prune_factor(f, max_factor_size); }
        }
    }

    template <typename T>
    void insert_pattern(float_type frequency,
                        const T&   t,
                        size_t     max_factor_size,
                        size_t     max_factor_width,
                        bool       estimate)
    {

        std::vector<std::pair<size_t, bool>> selection;
        selection.reserve(count(t));

        bool found_superset = false;
        factorize_statically(phi, t, [&](const auto& f, size_t i, bool from_singleton) {
            if (!found_superset)
            {
                if (is_subset(t, f.range) && !from_singleton)
                {
                    if (f.factor.itemsets.set.size() < max_factor_size)
                    {
                        auto& phi_i = phi.factors[i];
                        if (estimate) prune_factor_if(phi_i, max_factor_size);
                        phi_i.factor.insert(frequency, t, estimate);
                    }
                    found_superset = true;
                    return;
                }
                else
                {
                    selection.emplace_back(i, from_singleton);
                }
            }
        });

        if (selection.empty() || found_superset) { return; }

        factor_type next(dim);

        for (const auto& [i, s] : selection)
        {
            join_factors(next, s ? phi.singleton_factors[i] : phi.factors[i]);
        }

        if (count(next.range) > max_factor_width ||
            next.factor.itemsets.set.size() > max_factor_size)
        {
            // #ifndef NDEBUG
            //             throw std::domain_error{"pattern too large or factor is full"};
            // #endif
            return;
        }

        if (estimate) prune_factor_if(next, max_factor_size);
        next.factor.insert(frequency, t, estimate);

        for (auto [i, s] : selection)
        {
            if (!s) phi.factors[i].range.clear();
        }
        selection.clear();

        erase_empty_factors(phi);
        phi.factors.emplace_back(std::move(next));
    }

    template <typename T>
    void insert_pattern(float_type frequency, const T& t, bool estimate)
    {
        insert_pattern(frequency, t, max_factor_size, max_factor_width, estimate);
    }

    template <typename T>
    void insert(float_type frequency, const T& t, bool estimate)
    {
        if (is_singleton(t))
            insert_singleton(frequency, t, estimate);
        else
            insert_pattern(frequency, t, estimate);
    }

    template <typename T>
    bool is_allowed(const T& t, size_t max_factor_size, size_t max_factor_width) const
    {
        size_t total_size     = 0;
        size_t total_width    = 0;
        bool   found_superset = false;

        factorize_statically(phi, t, [&](const auto& f, size_t, bool) {
            if (is_subset(t, f.range))
            {
                found_superset = true;
                total_size     = f.factor.itemsets.set.size();
                total_width    = f.factor.singletons.set.size();
            }
            else if (!found_superset)
            {
                total_size += f.factor.itemsets.set.size();
                total_width += count(f.range);
            }
        });

        return total_size < max_factor_size && total_width < max_factor_width;
    }

    template <typename T>
    bool is_allowed(const T& t) const
    {
        return is_allowed(t, max_factor_size, max_factor_width);
    }
};

template <typename S, typename T, typename U, typename X, typename Visitor>
void factorize(const StaticFactorModel<S, T, U>& pr, const X& x, Visitor&& f)
{
    factorize_statically(pr.phi, x, std::forward<Visitor>(f));
}
template <typename U, typename V, typename X, typename Visitor>
void factorize_patterns(const StaticFactorModel<U, V>& pr, const X& x, Visitor&& f)
{
    thread_local itemset<typename StaticFactorModel<U, V>::pattern_type> y;
    y.clear();
    y.assign(x);

    factorize_patterns_statically(pr.phi.factors, y, f);
}
template <typename S, typename T, typename U, typename F = double>
void estimate_model(StaticFactorModel<S, T, U>& m, IterativeScalingSettings<F> const& opts = {})
{
    estimate_model(m.phi.factors, opts);
}

template <typename U, typename V, typename Underlying_Factor_Type = MaxEntFactor<U, V>>
struct ClassicStaticFactorModel
{
    using float_type            = V;
    using value_type            = V;
    using pattern_type          = U;
    using index_type            = std::size_t;
    using underlying_model_type = Underlying_Factor_Type;
    // using cache_type            = Cache<U, V>;
    using factor_type = Factor<underlying_model_type>;

    constexpr static bool enable_factor_pruning = false;

    explicit ClassicStaticFactorModel(size_t dim) : dim(dim) { init(dim); }
    ClassicStaticFactorModel() = default;

    size_t dim              = 0;
    size_t count_sets       = 0;
    size_t max_factor_size  = 5;
    size_t max_factor_width = 8;

    struct
    {
        std::vector<factor_type> factors;
    } phi;

    size_t num_itemsets() const { return count_sets; }
    size_t dimension() const { return dim; }
    size_t size() const { return num_itemsets() + dimension(); }

    void init(size_t dim)
    {
        this->dim = dim;
        clear();
        init_singletons(phi.factors, dim);
    }

    void clear() { phi.factors.clear(); }

    void insert_singleton(value_type frequency, const index_type element, bool estimate)
    {
        disc::insert_singleton(phi.factors, dim, frequency, element, estimate);
    }

    template <typename T>
    void insert_singleton(value_type frequency, const T& t, bool estimate)
    {
        insert_singleton(frequency, static_cast<index_type>(front(t)), estimate);
    }

    void prune_factor_if(factor_type& f, size_t max_factor_size)
    {
        if constexpr (enable_factor_pruning)
        {
            if (f.factor.itemsets.set.size() > 3) { prune_factor(f, max_factor_size); }
        }
    }

    template <typename T>
    void insert_pattern(value_type frequency,
                        const T&   t,
                        size_t     max_factor_size,
                        size_t     max_factor_width,
                        bool       estimate)
    {
        std::vector<size_t> selection;
        selection.reserve(count(t));

        for (size_t i = 0, n = phi.factors.size(); i < n; ++i)
        {
            auto& f = phi.factors[i];
            if (is_subset(t, f.range))
            {
                if (f.factor.itemsets.set.size() < max_factor_size)
                {
                    if (estimate) prune_factor_if(f, max_factor_size);
                    f.factor.insert(frequency, t, estimate);
                }
                return;
            }
            if (intersects(t, f.range)) selection.push_back(i);
        }

        if (selection.empty()) { return; }

        factor_type next(dim);

        for (const auto& s : selection) { join_factors(next, phi.factors[s]); }

        if (count(next.range) > max_factor_width ||
            next.factor.itemsets.set.size() > max_factor_size)
        {
            // #ifndef NDEBUG
            //             throw std::domain_error{"pattern too large or factor is full"};
            // #endif
            return;
        }

        if (estimate) prune_factor_if(next, max_factor_size);
        next.factor.insert(frequency, t, estimate);

        for (auto i : selection) { phi.factors[i].range.clear(); }
        selection.clear();

        erase_empty_factors();

        phi.factors.emplace_back(std::move(next));
        ++count_sets;
    }

    template <typename T>
    void insert_pattern(value_type frequency, const T& t, bool estimate)
    {
        insert_pattern(frequency, t, max_factor_size, max_factor_width, estimate);
    }

    template <typename T>
    void insert(value_type frequency, const T& t, bool estimate)
    {
        if (is_singleton(t))
            insert_singleton(frequency, t, estimate);
        else
            insert_pattern(frequency, t, estimate);
    }

    template <typename T>
    bool is_allowed(const T& t, size_t max_factor_size, size_t max_factor_width) const
    {
        size_t total_size  = 0;
        size_t total_width = 0;
        for (auto& f : phi.factors)
        {
            if (is_subset(t, f.range))
            {
                return f.factor.itemsets.set.size() < max_factor_size;
            }
            if (intersects(t, f.range))
            {
                // this works, because all factors are covering disjoint sets
                total_size += f.factor.itemsets.set.size();
                total_width += count(f.range);
                if (total_size >= max_factor_size || total_width >= max_factor_width)
                    return false;
            }
        }

        return true;
    }

    template <typename T>
    bool is_allowed(const T& t) const
    {
        return is_allowed(t, max_factor_size, max_factor_width);
    }

    underlying_model_type as_single_factor() const
    {
        factor_type next(dim);
        for (const auto& s : phi.factors) { join_factors(next, s); }
        return next.factor;
    }

    void erase_empty_factors()
    {
        phi.factors.erase(std::remove_if(phi.factors.begin(),
                                         phi.factors.end(),
                                         [](auto& f) { return f.range.empty(); }),
                          phi.factors.end());
        // for (size_t i = 0; i < factors.size();)
        // {
        //     if (factors[i].range.empty())
        //     {
        //         if (i != factors.size() - 1)
        //         {
        //             std::swap(factors[i], factors.back());
        //         }

        //         factors.pop_back();
        //     }
        //     else
        //     {
        //         ++i;
        //     }
        // }
    }
};
template <typename S, typename T, typename U, typename X, typename Visitor>
void factorize(const ClassicStaticFactorModel<S, T, U>& pr, const X& x, Visitor&& v)
{
    factorize_classic(pr.phi.factors, x, [&](const auto& f, size_t i) {
        v(f, i, f.factor.singletons.set.size() == 1);
    });
}
template <typename U, typename V, typename X, typename Visitor>
void factorize_patterns(const ClassicStaticFactorModel<U, V>& pr, const X& x, Visitor&& v)
{
    factorize_classic(pr.phi.factors, x, [&](const auto& f, size_t i) {
        if (f.factor.singletons.set.size() != 1) v(f, i);
    });
}
template <typename S, typename T, typename U, typename F = double>
void estimate_model(ClassicStaticFactorModel<S, T, U>& m,
                    IterativeScalingSettings<F> const& opts = {})
{
    estimate_model(m.phi.factors, opts);
}

template <typename U, typename X, typename Visitor>
void factorize_singletons(const Factorization<U>& phi, const X& x, Visitor&& v)
{
    foreach (x, [&](size_t i) { v(phi.singleton_factors[i], i); })
        ;
}
template <typename S, typename T, typename U, typename X, typename Visitor>
void factorize_singletons(const StaticFactorModel<S, T, U>& pr, const X& x, Visitor&& v)
{
    factorize_singletons(pr.phi, x, std::forward<Visitor>(v));
}

} // namespace disc
} // namespace sd