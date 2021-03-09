#pragma once

#include <desc/distribution/Transactions.hxx>
#include <desc/storage/Dataset.hxx>
#include <desc/storage/Itemset.hxx>

#include <cstddef>
#include <numeric>
#include <vector>

#include <optional>

namespace sd
{
namespace viva
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
        if (it != set.end())
        {
            it->frequency = label;
        }
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
        if (it != end(set))
        {
            return {it->probability};
        }
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
        if (it != set.end())
        {
            it->frequency = label;
        }
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
        if (it != set.end())
        {
            return {it->probability};
        }
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
    void insert_pattern(float_type label, const T& t, bool estimate = false)
    {
        itemsets.insert(label, t);
        itemsets.num_singletons = singletons.set.size();
        if (estimate)
        {
            estimate_model(*this);
        }
    }

    template <typename T>
    void insert_singleton(float_type label, T&& t, bool estimate = false)
    {
        singletons.insert(label, t);
        itemsets.num_singletons = singletons.set.size();
        if (estimate)
        {
            estimate_model(*this);
        }
    }

    template <typename T>
    void insert(float_type label, const T& t, bool estimate = false)
    {
        if (is_singleton(t))
        {
            insert_singleton(label, t);
        }
        else
        {
            insert_pattern(label, t);
        }

        if (estimate)
            estimate_model(*this);
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
    if (is_singleton(t))
        return contains_singleton(m.singletons, t);
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
    if (is_singleton(t))
        return erase_if(m.singletons, t);
    return erase_if(m.itemsets, t);
}

} // namespace viva
} // namespace sd