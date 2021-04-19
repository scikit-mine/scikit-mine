#pragma once

#include <desc/Settings.hxx>
#include <desc/distribution/Distribution.hxx>
#include <desc/storage/Dataset.hxx>
#include <ndarray/ndarray.hxx>
#include <vector>

namespace sd
{
namespace disc
{

using AssignmentMatrix = std::vector<sparse_dynamic_bitset<size_t>>;
size_t trace(const AssignmentMatrix& a)
{
    return std::accumulate(
        begin(a), end(a), size_t(), [&](auto acc, const auto& x) { return acc + count(x); });
}

template <typename T>
struct EncodingLength
{
    T    of_data{std::numeric_limits<T>::max()};
    T    of_model{0};
    auto objective() const { return of_data + of_model; }
};

template <typename Pattern_Type, typename Float_Type, typename Distribution_Type>
struct Trait
{
    using pattern_type      = Pattern_Type;
    using float_type        = Float_Type;
    using distribution_type = Distribution_Type;
    using size_type         = size_t;
};

using DefaultTrait = Trait<tag_dense, double, MaxEntDistribution<tag_dense, double>>;

template <typename Trait>
struct Composition
{
    using pattern_type      = typename Trait::pattern_type;
    using float_type        = typename Trait::float_type;
    using distribution_type = typename Trait::distribution_type;
    using size_type         = typename Trait::size_type;
    using tid_container     = long_storage_container<typename Trait::pattern_type>;

    PartitionedData<pattern_type>  data;
    Dataset<pattern_type>          summary;
    AssignmentMatrix               assignment;
    sd::ndarray<float_type, 2>     confidence;
    sd::ndarray<float_type, 2>     frequency;
    // EncodingLength<float_type>     encoding;
    // EncodingLength<float_type>     initial_encoding;
    std::vector<distribution_type> models;
    std::vector<tid_container> masks;
};

template <typename T>
bool check_invariant(const Composition<T>& comp)
{
    return (comp.frequency.extent(0) == comp.summary.size() &&
            comp.frequency.extent(1) >= comp.data.num_components() &&
            comp.confidence.extent(0) == comp.summary.size() &&
            comp.confidence.extent(1) == comp.data.num_components() &&
            comp.assignment.size() == comp.data.num_components() &&
            comp.models.size() == comp.assignment.size()
            // comp.summary.size() >= comp.data.dim &&
    );
}

template <typename S>
auto make_distribution(S const& c, Config const& cfg)
{
    using distribution_type = typename S::distribution_type;
    return distribution_type(c.data.dim, c.data.size(), cfg);
}

template <typename Trait>
auto construct_component_masks(const Composition<Trait>& c)
{
    using tid_container = long_storage_container<typename Trait::pattern_type>;

    assert(check_invariant(c));

    std::vector<tid_container> masks;

    if (c.data.empty()) return masks;

    masks.resize(c.data.num_components(), tid_container{c.data.size()});
    for (size_t s = 0, row = 0, n = c.data.num_components(); s < n; ++s)
    {
        masks[s].clear();
        for ([[maybe_unused]] const auto& x : c.data.subset(s))
        {
            masks[s].insert(row);
            ++row;
        }
    }

    return masks;
}

} // namespace disc
} // namespace sd