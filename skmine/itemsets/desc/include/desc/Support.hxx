#pragma once

#include <desc/Component.hxx>
#include <desc/Composition.hxx>
#include <desc/storage/Dataset.hxx>
#include <ndarray/ndarray.hxx>

namespace sd
{
namespace disc
{

template <typename P, typename Data>
size_t support(const P& x, const Data& es)
{
    return std::count_if(
        es.begin(), es.end(), [&](const auto& e) { return is_subset(x, point(e)); });
}
template <typename T, typename P, typename E>
T frequency(const P& x, const E& es)
{
    return static_cast<T>(support(x, es)) / es.size();
}

// template <typename P, typename Data>
// bool contains_geq(const P& x, const Data& es, size_t bound)
// {
//     size_t c = 0;
//     for (const auto& e : es)
//     {
//         if (is_subset(x, point(e)) && ++c >= bound)
//         {
//             return true;
//         }
//     }
//     return false;
// }

// template <typename P, typename Data>
// bool contains_any_superset(const P& x, const Data& es)
// {
//     return std::any_of(
//         es.begin(), es.end(), [&](const auto& e) { return is_subset(x, point(e)); });
// }

// template <typename P, typename Data>
// bool contains_any(const P& x, const Data& es)
// {
//     return std::any_of(es.begin(), es.end(), [&](const auto& e) { return equal(point(e), x);
//     });
// }

template <typename DataType>
std::vector<size_t> compute_singleton_supports(size_t dim, const DataType& data)
{
    std::vector<size_t> support(dim);
    for (size_t i = 0; i < data.size(); ++i)
    {
        assert(i < data.size());
        foreach (data.point(i), [&](auto j) {
            assert(j < dim);
            support[j]++;
        })
            ;
    }
    return support;
}

template <typename DataType, typename Summary>
void insert_missing_singletons(const DataType& data, Summary& summary)
{
    // using value_type = typename Summary::label_type;

    itemset<tag_dense> set(std::max(data.dim, summary.dim));

    summary.dim = std::max(data.dim, summary.dim);

    for (const auto& x : summary)
        if (is_singleton(point(x))) set.insert(front(point(x)));

    if (set.count() == data.dim) { return; }

    // auto support = compute_singleton_supports(data.dim, data);
    // assert(support.size() == data.dim);

    storage_container<typename Summary::pattern_type> x;

    for (size_t i = 0; i < data.dim; ++i)
    {
        if (!set.test(i)) // && support[i] > 0)
        {
            // auto fr = static_cast<value_type>(support[i]) / data.size();
            // summary.insert(fr, i);
            x.clear();
            x.insert(i);
            summary.insert(x);
        }
    }
}

template <typename Trait>
void compute_frequency_matrix_column(Composition<Trait>& c, const size_t comp_index)
{
    auto& data    = c.data;
    auto& summary = c.summary;
    auto& fr      = c.frequency;

    for (size_t i = 0; i < fr.extent(0); ++i) fr(i, comp_index) = 0;

    auto component = data.subset(comp_index);
    for (const auto& x : component)
    {
        for (size_t i = 0; i < summary.size(); ++i)
        {
            if (is_subset(summary.point(i), point(x))) { fr(i, comp_index) += 1; }
        }
    }

    for (size_t i = 0; i < summary.size(); ++i) { fr(i, comp_index) /= component.size(); }
}

template <typename Trait>
void compute_frequency_matrix(Composition<Trait>& c)
{

    size_t n_cols = c.data.num_components();

    // c.frequency = sd::ndarray<typename Trait::float_type, 2>({c.summary.size(), n_cols});

    c.frequency.clear();
    c.frequency.resize(sd::layout<2>({c.summary.size(), n_cols}), 0.0);

    for (size_t j = 0; j < n_cols; ++j) { compute_frequency_matrix_column(c, j); }
}

template <typename Trait>
void compute_frequency_matrix_column(Component<Trait>& c)
{
    auto& data = c.data;
    auto& s    = c.summary;
    auto& fr   = c.frequency;

    fr.assign(c.summary.size(), 0);

    for (const auto& x : data)
    {
        for (size_t i = 0; i < s.size(); ++i)
        {
            if (is_subset(s.point(i), point(x))) { fr[i] += 1; }
        }
    }

    for (size_t i = 0; i < s.size(); ++i) { fr[i] /= data.size(); }
}

// template <typename T, typename S>
// void compute_frequency_matrix(const PartitionedData<S>&   data,
//                               const LabeledDataset<T, S>& summary,
//                               sd::ndarray<T, 2>&          fr)
// {
//     size_t n_cols = data.num_components();
//     if (n_cols > 1)
//         n_cols += 1;

//     fr.clear();
//     fr.resize(sd::layout<2>({summary.size(), n_cols}), 0.0);

//     if (fr.extent(1) == 1)
//     {
//         std::copy_n(summary.labels().begin(), summary.size(), fr[0].begin());
//     }
//     else
//     {
//         for (size_t j = 0; j < n_cols - 1; ++j)
//         {
//             compute_frequency_matrix_column(data, summary, j, fr);
//         }
//         for (size_t i = 0; i < summary.size(); ++i)
//         {
//             fr(i, n_cols - 1) = summary.label(i);
//         }
//     }
// }

} // namespace disc
} // namespace sd