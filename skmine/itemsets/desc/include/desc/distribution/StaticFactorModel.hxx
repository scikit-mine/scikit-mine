#pragma once

// #include <desc/distribution/FactorPruning.hxx>
#include <desc/distribution/MaxEntFactor.hxx>
#include <desc/storage/Itemset.hxx>

#ifndef NDEBUG
#include <exception>
#endif

namespace sd
{
namespace viva
{

template <typename Underlying_Factor_Type>
struct Factor
{
    using pattern_type = typename Underlying_Factor_Type::pattern_type;
    explicit Factor(size_t dim) : factor(dim) { range.reserve(dim); }
    sd::disc::itemset<pattern_type> range;
    Underlying_Factor_Type          factor;
};

template <typename U>
void join_factors(Factor<U>& f, const Factor<U>& g)
{
    for (auto& t : g.factor.itemsets.set)
        f.factor.insert_pattern(t.frequency, t.point);

    for (auto& t : g.factor.singletons.set)
        f.factor.insert_singleton(t.frequency, t.element);

    f.range.insert(g.range);
}

template <typename U, typename V, typename Underlying_Factor_Type = MaxEntFactor<U, V>>
struct StaticFactorModel
{
    using float_type            = V;
    using value_type            = V;
    using pattern_type          = U;
    using index_type            = std::size_t;
    using underlying_model_type = Underlying_Factor_Type;
    // using cache_type            = Cache<U, V>;
    using factor_type = Factor<underlying_model_type>;

    // constexpr static bool enable_factor_pruning = false;

    explicit StaticFactorModel(size_t dim) : dim(dim) { init(dim); }
    StaticFactorModel() = default;

    size_t dim              = 0;
    size_t count_sets       = 0;
    size_t max_factor_size  = 5;
    size_t max_factor_width = 8;

    std::vector<factor_type> factors;

    size_t num_itemsets() const { return count_sets; }
    size_t dimension() const { return dim; }
    size_t size() const { return num_itemsets() + dimension(); }

    void init(size_t dim)
    {
        this->dim = dim;
        factors.clear();
        factors.resize(dim, factor_type(dim));

        for (size_t i = 0; i < dim; ++i)
        {
            factors[i].range.insert(i);
            factors[i].factor.insert_singleton(0.5, i);
            estimate_model(factors[i].factor);
        }
    }

    void clear() { factors.clear(); }

    void insert_singleton(value_type frequency, const index_type element, bool estimate = false)
    {
        bool add_new = true;
        for (auto& f : factors)
        {
            if (is_subset(element, f.range))
            {
                f.factor.insert_singleton(frequency, element, estimate);
                if (is_singleton(f.range))
                {
                    add_new = false;
                }
            }
        }

        if (add_new)
        {
            auto& f = factors.emplace_back(dim);
            f.range.insert(element);
            f.factor.insert_singleton(frequency, element, estimate);
        }
    }

    template <typename T>
    void insert_singleton(value_type frequency, const T& t, bool estimate = false)
    {
        insert_singleton(frequency, static_cast<index_type>(front(t)), estimate);
    }

    // void prune_factor_if(factor_type& f, size_t max_factor_size)
    // {
    //     if constexpr (enable_factor_pruning)
    //     {
    //         if (f.factor.itemsets.set.size() > 3)
    //         {
    //             prune_factor(f, max_factor_size);
    //         }
    //     }
    // }

    template <typename T>
    void insert_pattern(value_type frequency,
                        const T&   t,
                        size_t     max_factor_size,
                        size_t     max_factor_width,
                        bool       estimate)
    {
        std::vector<size_t> selection;
        selection.reserve(count(t));

        for (size_t i = 0, n = factors.size(); i < n; ++i)
        {
            auto& f = factors[i];
            if (is_subset(t, f.range))
            {
                if (f.factor.itemsets.set.size() < max_factor_size)
                {
                    // if (estimate)
                    // {
                    //     prune_factor_if(f, max_factor_size);
                    // }
                    f.factor.insert(frequency, t, estimate);
                }
                return;
            }
            if (intersects(t, f.range))
                selection.push_back(i);
        }

        if (selection.empty())
        {
            return;
        }

        factor_type next(dim);

        for (const auto& s : selection)
        {
            join_factors(next, factors[s]);
        }

        // if (estimate)
        // {
        //     prune_factor_if(next, max_factor_size);
        // }

        if (count(next.range) > max_factor_width ||
            next.factor.itemsets.set.size() > max_factor_size)
        {
            // #ifndef NDEBUG
            //             throw std::domain_error{"pattern too large or factor is full"};
            // #endif
            return;
        }

        next.factor.insert(frequency, t, estimate);

        for (auto i : selection)
        {
            factors[i].range.clear();
        }
        selection.clear();

        erase_empty_factors();

        factors.emplace_back(std::move(next));
        ++count_sets;
    }

    template <typename T>
    void insert_pattern(value_type frequency, const T& t, bool estimate = false)
    {
        insert_pattern(frequency, t, max_factor_size, max_factor_width, estimate);
    }

    template <typename T>
    void insert(value_type frequency, const T& t, bool estimate = false)
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
        for (auto& f : factors)
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
        for (const auto& s : factors)
        {
            join_factors(next, s);
        }
        return next.factor;
    }

    void erase_empty_factors()
    {
        factors.erase(std::remove_if(factors.begin(),
                                     factors.end(),
                                     [](auto& f) { return f.range.empty(); }),
                      factors.end());
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

} // namespace viva
} // namespace sd