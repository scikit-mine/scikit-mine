#pragma once

#include <desc/Composition.hxx>
#include <desc/Settings.hxx>
// #include <desc/Support.hxx>
#include <desc/distribution/Distribution.hxx>
#include <desc/storage/Dataset.hxx>

namespace sd::disc
{

template <typename Trait>
struct Component
{
    using pattern_type      = typename Trait::pattern_type;
    using float_type        = typename Trait::float_type;
    using distribution_type = typename Trait::distribution_type;
    using size_type         = typename Trait::size_type;

    Dataset<pattern_type> data;
    // LabeledDataset<float_type, pattern_type> summary;
    Dataset<pattern_type>      summary;
    std::vector<float_type>    confidence;
    std::vector<float_type>    frequency;
    // EncodingLength<float_type> encoding;
    // EncodingLength<float_type> initial_encoding;
    distribution_type          model;

    // template <
    //     typename DATA,
    //     typename = std::enable_if_t<!std::is_same_v<std::decay_t<DATA>, Component<Trait>>>>
    // explicit Component(DATA&& d) : data(std::forward<DATA>(d))
    // {
    // }

    Component()                 = default;
    Component(Component&&)      = default;
    Component(const Component&) = default;
    Component& operator=(const Component&) = default;
    Component& operator=(Component&&) = default;

    // explicit operator Composition<Trait>() const
    // {
    //     Composition<Trait> c;

    //     c.data.reserve(data.size());
    //     for (size_t i = 0; i < data.size(); ++i) { c.data.insert(data.point(i), 0, i); }
    //     c.data.group_by_label();

    //     c.encoding         = encoding;
    //     c.initial_encoding = initial_encoding;
    //     c.summary          = summary;
    //     c.models           = {model};

    //     auto& a = c.assignment.emplace_back();
    //     for (size_t i = 0; i < summary.size(); ++i) { a.insert(i); }

    //     compute_frequency_matrix(c);
    //     return c;
    // }
};

// template <typename Trait>
// typename Trait::distribution_type& initialize_model(Component<Trait>& c, const Config& cfg =
// {})
// {
//     auto& data    = c.data;
//     auto& summary = c.summary;

//     insert_missing_singletons(data, summary);

//     data.dim = std::max(data.dim, summary.dim);

//     c.model = make_distribution(c, cfg);

//     assert(c.model.model.dim == data.dim);
//     assert(c.model.model.factors.size() == data.dim);

//     auto& pr = c.model;

//     for (const auto& i : summary)
//     {
//         if (is_singleton(point(i))) { pr.insert_singleton(label(i), point(i), false); }
//     }

//     for (const auto& i : summary)
//     {
//         if (!is_singleton(point(i)) && pr.is_allowed(point(i)))
//         {
//             pr.insert(label(i), point(i), false);
//         }
//     }

//     estimate_model(pr);

//     return pr;
// }

} // namespace sd::disc