#pragma once

#include <desc/distribution/BitPermutation.hxx>
#include <desc/distribution/IncrementBitset.hxx>
#include <desc/storage/Itemset.hxx>

#include <optional>

namespace sd
{
namespace disc
{

template <typename Fn>
void iterate_powerset_short(size_t n, Fn&& fn)
{
    const auto one   = size_t(1);
    const auto power = (one << n);
    for (size_t i = power; i-- > 0;)
    {
        std::forward<Fn>(fn)(i);
    }
}

template <typename Fn>
void iterate_powerset_long(size_t n, Fn&& fn)
{
    for (disc::itemset<disc::tag_dense> i(n, false); true; increment(i))
    {
        std::forward<Fn>(fn)(i);
        if (count(i) == n)
            break;
    }
}

template <typename Fn>
void iterate_powerset(size_t n, Fn&& fn)
{
    if (n < std::numeric_limits<size_t>::digits)
    {
        iterate_powerset_short(n, std::forward<Fn>(fn));
    }
    else
    {
        iterate_powerset_long(n, std::forward<Fn>(fn));
    }
}

template <typename U, typename V>
struct Block
{
    using count_type = V;

    count_type       value{0};
    count_type       count{0};
    disc::itemset<U> cover;

    bool operator<(Block const& rhs) const { return count < rhs.count; }
};

template <typename model_type, typename block_container_type>
size_t generate_blocks_long(size_t dim, model_type const& m, block_container_type& blocks)
{
    using float_type   = typename model_type::float_type;
    using pattern_type = typename model_type::pattern_type;

    const auto size = m.size();

    if (size == 0 || dim == 0)
    {
        const auto k = std::exp2(float_type(dim));

        blocks.resize(1);
        blocks[0].value = k;
        blocks[0].count = k;
        // blocks[0].cover.resize(dim);
        blocks[0].cover.clear();

        // blocks       = {{k, k, disc::itemset<pattern_type>(dim)}};

        return 1;
    }

    thread_local disc::itemset<pattern_type> cover;
    cover.reserve(dim);
    cover.clear();

    blocks.clear();
    for (disc::itemset<disc::tag_dense> i(size, false); true; increment(i))
    {
        cover.clear();

        foreach(i, [&](size_t j) { cover.insert(m.point(j)); });

        const auto k = std::exp2(float_type(dim) - float_type(count(cover)));
        assert(k > 0);
        blocks.push_back({k, k, cover});

        if (count(i) == size)
            break;
    }
    std::sort(blocks.begin(), blocks.end());
    auto it = std::unique(blocks.begin(), blocks.end(), [](const auto& a, const auto& b) {
        return equal(a.cover, b.cover);
    });
    blocks.erase(it, blocks.end());
    return blocks.size();
}

template <typename model_type, typename block_container_type>
void generate_blocks_and_counts_long(size_t                dim,
                                     model_type const&     model,
                                     block_container_type& blocks)
{
    auto n = generate_blocks_long(dim, model, blocks);
    if (n <= 1)
        return;

    for (size_t i = 0; i < n; ++i)
    {
        auto& b = blocks[i];
        b.value = b.count;
        for (size_t j = 0; j < i; ++j)
        {
            auto& a = blocks[j];
            if (is_subset(b.cover, a.cover))
            {
                if (count(b.cover) == count(a.cover) && equal(a.cover, b.cover))
                {
                    b.value = 0;
                    b.cover.clear();
                    break;
                }
                assert(b.value - a.value >= 0);
                b.value -= a.value;
            }
        }
    }
}

template <typename model_type, typename block_container_type>
size_t generate_blocks_and_counts(size_t dim, model_type const& m, block_container_type& blocks)
{

    assert(dim < std::numeric_limits<size_t>::digits);
    using float_type = typename model_type::float_type;
    // using pattern_type = typename model_type::pattern_type;
    // using storage_type = disc::itemset<pattern_type>;

    const auto size  = m.size();
    const auto width = static_cast<float_type>(dim);

    if (size == 0 || dim == 0)
    {
        // blocks       = {{k, k, disc::itemset<pattern_type>(dim)}};

        if (blocks.size() < 1)
        {
            blocks.resize(1);
        }

        auto& block = blocks.front();

        block.cover.clear();

        const auto k = std::exp2(width);
        block.value  = k;
        block.count  = k;

        return 1;
    }

    const size_t part_size = std::exp2(m.size());
    // blocks.clear();
    if (blocks.size() < part_size)
    {
        blocks.resize(part_size);
    }

    size_t index = 0;

    disc::permute_all(m.size(), [&](size_t i) {
        auto& block = blocks[index];

        block.cover.clear();
        block.cover.reserve(dim);

        foreach(i, [&](size_t j) { block.cover.insert(m.point(j)); });

        const auto cnt_block  = count(block.cover);
        const auto cover_size = static_cast<float_type>(cnt_block);
        const auto k          = std::exp2(width - cover_size);
        block.value           = k;
        block.count           = k;

        assert(width >= cover_size);
        assert(k > 0);
        assert(!std::isnan(k));

        for (size_t prev = 0; prev < index; ++prev)
        {
            if (is_subset(block.cover, blocks[prev].cover))
            {
                // if (cnt_block == count(blocks[prev].cover))
                if (block.count == blocks[prev].count)
                {
                    block.value = 0;
                    block.cover.clear();
                    break;
                }
                else
                {
                    block.value -= blocks[prev].value;
                    assert(block.value >= 0);
                    assert(floor(block.value) == block.value);
                }
            }
        }
        if (block.value != 0)
        {
            ++index;
        }
    });

    return index;
}

template <typename model_type, typename block_container_type>
size_t compute_counts(size_t dim, model_type const& m, block_container_type& blocks)
{
    // if (dim < std::numeric_limits<size_t>::digits)
    // {
    return generate_blocks_and_counts(dim, m, blocks);
    // }
    // else
    // {
    //     generate_blocks_and_counts_long(dim, m, blocks);
    //     return blocks.size();
    // }
}

} // namespace disc
} // namespace sd
