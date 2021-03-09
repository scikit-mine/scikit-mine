#pragma once

#include <desc/distribution/InferProbabilities.hxx>
#include <desc/distribution/MaxEntFactor.hxx>

namespace sd::viva
{

template <typename factor_type>
void prune_factor(factor_type& next, size_t max_factor_size)
{
    using float_type = std::decay_t<decltype(next.factor.singletons.set.front().frequency)>;

    if (next.factor.itemsets.set.size() == 0)
    {
        estimate_model(next.factor);
        return;
    }

    factor_type replacement(next.factor.singletons.dim);
    replacement.factor.singletons              = next.factor.singletons;
    replacement.factor.itemsets.num_singletons = replacement.factor.singletons.size();
    replacement.range.insert(next.range);

    estimate_model(replacement.factor);

    auto& xs = next.factor.itemsets.set;

    // small_bitset<size_t, 1> in_use;
    thread_local sd::disc::itemset<sd::disc::tag_dense> in_use; in_use.clear();

    for (size_t count = 0, n = xs.size(); count <= max_factor_size;)
    {
        std::pair best{n, float_type(0)};
        for (size_t j = n; j-- > 0;)
        {
            if (in_use.test(j))
                continue;

            auto p = expectation(replacement.factor, xs[j].point);
            auto q = xs[j].frequency;
            auto g = std::log2(q / p); // assignment_score

            if (g > best.second)
            {
                best = {j, g};
            }
        }

        if (best.first >= n)
            break;

        if (best.second > 0.0)
        {
            ++count;
            in_use.insert(best.first);
            replacement.factor.insert(xs[best.first].frequency, xs[best.first].point, true);
        }
        else
        {
            break;
        }
    }
    next = std::move(replacement);
    // assert(next.factor.itemsets.set.size() > 0);
}

} // namespace sd::viva