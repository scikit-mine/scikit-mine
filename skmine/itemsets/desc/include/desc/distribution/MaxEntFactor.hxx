#pragma once

#include <desc/distribution/Transactions.hxx>
#include <desc/storage/Dataset.hxx>
#include <desc/storage/Itemset.hxx>

#include <cstddef>
#include <numeric>
#include <optional>
#include <vector>

namespace sd
{
namespace disc
{

template <typename U, typename V>
struct SingletonModel
{
    using float_type   = V;
    using pattern_type = U;

    using index_type = std::size_t;

    struct singleton_storage
    {
        index_type element;
        float_type frequency;
        float_type theta       = 1;
        float_type probability = 0.5;
    };

    std::vector<singleton_storage> set;
    mutable disc::itemset<U>       buffer;

    float_type theta0 = 1;
    size_t     dim    = 0;

    auto&       coefficient(size_t i) { return set[i].theta; }
    auto&       normalizer() { return theta0; }
    const auto& normalizer() const { return theta0; }
    auto&       probability(size_t i) { return set[i].probability; }
    const auto& frequency(size_t i) const { return set[i].frequency; }
    const auto& probability(size_t i) const { return set[i].probability; }
    const auto& coefficient(size_t i) const { return set[i].theta; }
    const auto& point(size_t i) const
    {
        buffer.clear();
        buffer.insert(set[i].element);
        return buffer;
    }

    void insert(float_type label, index_type element)
    {
        auto it = std::find_if(
            set.begin(), set.end(), [element](const auto& i) { return i.element == element; });
        if (it != set.end()) { it->frequency = label; }
        else
        {
            set.push_back({element, label});
        }
    }

    template <typename T>
    void insert(float_type label, const T& t)
    {
        set.push_back({static_cast<index_type>(front(t)), label});
    }

    size_t dimension() const { return dim; }
    size_t size() const { return set.size(); }

    template <typename Pattern_Type>
    std::optional<float_type> get_precomputed_expectation(const Pattern_Type& x) const
    {
        auto element = front(x);
        auto it      = std::find_if(
            begin(set), end(set), [element](const auto& x) { return x.element == element; });
        if (it != end(set)) { return {it->probability}; }
        else
            return std::nullopt;
    }
};

template <typename U, typename V>
struct ItemsetStorage
{
    using float_type = V;

    disc::itemset<U> point{};
    float_type       frequency   = 0;
    float_type       theta       = 1;
    float_type       probability = 0.5;
};

template <typename U, typename V>
struct ItemsetModel
{
    using float_type   = V;
    using pattern_type = U;

    using itemset_storage = ItemsetStorage<U, V>;

    std::vector<itemset_storage> set;

    float_type       theta0         = 1;
    size_t           dim            = 0;
    size_t           num_singletons = 1;
    disc::itemset<U> buffer;

    template <typename T>
    void insert(float_type label, const T& t)
    {
        buffer.assign(t);
        auto it = std::find_if(
            set.begin(), set.end(), [&](const auto& i) { return equal(i.point, buffer); });
        if (it != set.end()) { it->frequency = label; }
        else
        {
            set.push_back({buffer, label, 1});
            // update_partitions();
        }
    }

    // void update_partitions() { compute_counts(width(), *this, partitions); }

    size_t width() const { return num_singletons; }
    size_t dimension() const { return dim; }
    size_t size() const { return set.size(); }

    auto&       coefficient(size_t i) { return set[i].theta; }
    auto&       normalizer() { return theta0; }
    const auto& normalizer() const { return theta0; }
    auto&       probability(size_t i) { return set[i].probability; }
    const auto& frequency(size_t i) const { return set[i].frequency; }
    const auto& probability(size_t i) const { return set[i].probability; }
    const auto& point(size_t i) const { return set[i].point; }
    const auto& coefficient(size_t i) const { return set[i].theta; }

    decltype(auto) operator[](size_t i) const { return point(i); }
    bool           empty() const { return set.empty(); }

    template <typename Pattern_Type>
    std::optional<float_type> get_precomputed_expectation(const Pattern_Type& x) const
    {
        const auto it = std::find_if(
            set.begin(), set.end(), [&](const auto& s) { return equal(s.point, x); });
        if (it != set.end()) { return {it->probability}; }
        else
            return std::nullopt;
    }
};

template <typename U, typename V>
struct MaxEntFactor
{
    using float_type   = V;
    using pattern_type = U;

    SingletonModel<U, V> singletons;
    ItemsetModel<U, V>   itemsets;

    explicit MaxEntFactor(size_t w = 0)
    {
        singletons.dim          = w;
        itemsets.dim            = w;
        itemsets.num_singletons = 1;
    }

    auto& coefficient(size_t i) const
    {
        return i < singletons.set.size() ? singletons.coefficient(i)
                                         : itemsets.coefficient(i - singletons.set.size());
    }
    auto& coefficient(size_t i)
    {
        return i < singletons.set.size() ? singletons.coefficient(i)
                                         : itemsets.coefficient(i - singletons.set.size());
    }
    auto&       normalizer() { return itemsets.normalizer(); }
    const auto& normalizer() const { return itemsets.normalizer(); }
    auto        frequency(size_t i) const
    {
        return i < singletons.set.size() ? singletons.frequency(i)
                                         : itemsets.frequency(i - singletons.set.size());
    }
    auto& probability(size_t i)
    {
        return i < singletons.set.size() ? singletons.probability(i)
                                         : itemsets.probability(i - singletons.set.size());
    }
    auto& probability(size_t i) const
    {
        return i < singletons.set.size() ? singletons.probability(i)
                                         : itemsets.probability(i - singletons.set.size());
    }
    const auto& point(size_t i)
    {
        return i < singletons.set.size() ? singletons.point(i)
                                         : itemsets.point(i - singletons.set.size());
    }
    bool   is_pattern_known(size_t i) const { return i >= singletons.set.size(); }
    size_t size() const { return singletons.set.size() + itemsets.set.size(); }
    size_t dimension() const { return singletons.dim; }

    template <typename T>
    bool is_allowed(const T&) const
    {
        return true;
    }

    template <typename T>
    void insert_pattern(float_type label, const T& t, bool estimate)
    {
        itemsets.insert(label, t);
        itemsets.num_singletons = singletons.set.size();
        if (estimate) { estimate_model(*this); }
    }

    template <typename T>
    void insert_singleton(float_type label, T&& t, bool estimate)
    {
        singletons.insert(label, t);
        itemsets.num_singletons = singletons.set.size();
        if (estimate) { estimate_model(*this); }
    }

    template <typename T>
    void insert(float_type label, const T& t, bool estimate)
    {
        if (is_singleton(t)) { insert_singleton(label, t, estimate); }
        else
        {
            insert_pattern(label, t, estimate);
        }
    }

    template <typename Pattern_Type>
    std::optional<float_type> get_precomputed_expectation(const Pattern_Type& x) const
    {
        return is_singleton(x) ? singletons.get_precomputed_expectation(x)
                               : itemsets.get_precomputed_expectation(x);
    }
};

template <typename S, typename T, typename U>
bool contains_pattern(const ItemsetModel<S, T>& m, const U& t)
{
    return std::any_of(
        m.set.begin(), m.set.end(), [&](const auto& i) { return equal(i.point, t); });
}

template <typename S, typename T, typename U>
bool contains_pattern(const MaxEntFactor<S, T>& m, const U& t)
{
    return contains_pattern(m.itemsets, t);
}

template <typename S, typename T, typename U>
bool contains_singleton(const SingletonModel<S, T>& m, const U& t)
{
    return std::any_of(
        m.set.begin(), m.set.end(), [&](const auto& i) { return is_subset(i.element, t); });
}

template <typename S, typename T, typename U>
bool contains(const MaxEntFactor<S, T>& m, const U& t)
{
    if (is_singleton(t)) return contains_singleton(m.singletons, t);
    return contains_pattern(m.itemsets, t);
}

template <typename S, typename T, typename U>
bool erase_if(ItemsetModel<S, T>& m, const U& t)
{
    auto pos = std::find_if(
        m.set.begin(), m.set.end(), [&](const auto& x) { return equal(x.point, t); });

    if (pos != m.set.end())
    {
        m.set.erase(pos);
        return true;
    }
    return false;
}
template <typename S, typename T, typename U>
bool erase_if(SingletonModel<S, T>& m, const U& t)
{
    auto pos = std::find_if(m.set.begin(), m.set.end(), [elem = front(t)](const auto& x) {
        return x.element == elem;
    });

    if (pos != m.set.end())
    {
        m.set.erase(pos);
        return true;
    }
    return false;
}

template <typename S, typename T, typename U>
bool erase_if(MaxEntFactor<S, T>& m, const U& t)
{
    if (is_singleton(t)) return erase_if(m.singletons, t);
    return erase_if(m.itemsets, t);
}

template <typename pattern_type, typename float_type, typename query_type>
auto probability(ItemsetModel<pattern_type, float_type> const& c, const query_type& t)
{
    float_type acc = c.theta0;
    for (size_t i = 0, l = c.set.size(); i < l; ++i)
    {
        if (is_subset(c.set[i].point, t)) { acc *= c.set[i].theta; }
    }
    return acc;
}

template <typename pattern_type, typename float_type, typename query_type>
auto probability(SingletonModel<pattern_type, float_type> const& c, query_type const& t)
{
    float_type acc = c.theta0;
    for (size_t i = 0, l = c.set.size(); i < l; ++i)
    {
        if (is_subset(c.set[i].element, t)) { acc *= c.set[i].theta; }
    }
    return acc;
}

template <typename pattern_type, typename float_type, typename query_type>
auto probability(MaxEntFactor<pattern_type, float_type> const& c, query_type const& t)
{
    return probability(c.itemsets, t) * probability(c.singletons, t);
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
                   itemset<pattern_type> const&                  additional_pattern)
        : underlying_model(underlying_model), additional_pattern(additional_pattern)
    {
    }

    size_t size() const { return underlying_model.size() + 1; }

    const auto& point(size_t i) const
    {
        if (i < underlying_model.size()) { return underlying_model.point(i); }
        else
        {
            return additional_pattern;
        }
    }

    ItemsetModel<pattern_type, float_type> const& underlying_model;
    itemset<pattern_type> const&                  additional_pattern;
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
    return disc::compute_counts(dimension_of_factor(model, x), augment_model(model, x), b);
}

template <typename S, typename T, size_t N = 13>
struct Temp_Partition_Buffer
{
    Temp_Partition_Buffer()
    {
        for (size_t i = 0; i < N; ++i) { blocks_of_size[i].resize(1 << i); }
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
        return disc::compute_counts(
            dimension_of_factor(model, x), augment_model(model, x), blocks);
    }
    else
    {
        return disc::compute_counts(dimension_of_factor(model, x), model.itemsets, blocks);
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
        if (!is_subset(s.element, t)) { p *= float_type(1.0) - s.probability; }
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
        if (!is_subset(s.element, t)) { p += std::log2(float_type(1.0) - s.probability); }
    }
    return p;
}

// template <typename Model, typename query_type>
// auto expectation_generalized_set(Model const& m, query_type const& t)
// {
//     return expectation(m, t) * probability_of_absent_items(m, t);
// }

} // namespace disc
} // namespace sd