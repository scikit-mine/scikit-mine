#pragma once

#include <desc/storage/Dataset.hxx>
#include <desc/storage/Itemset.hxx>

#include <algorithm>
#include <optional>
#include <vector>

#if defined(HAS_EXECUTION_POLICIES)
#include <execution>
#include <tbb/iterators.h>
#endif

namespace sd
{
namespace disc
{

template <typename S, typename T>
struct SlimCandidate
{
    using pattern_type = S;
    itemset<pattern_type>                pattern;
    long_storage_container<pattern_type> row_ids;
    size_t                               support = 0;
    T                                    score   = 0;
};

template <typename S, typename T>
void swap(SlimCandidate<S, T>& a, SlimCandidate<S, T>& b)
{
    using sd::swap;
    using std::swap;

    swap(a.support, b.support);
    swap(a.score, b.score);
    swap(a.pattern, b.pattern);
    swap(a.row_ids, b.row_ids);
}

template <typename S, typename T>
void join(SlimCandidate<S, T>& next, const SlimCandidate<S, T> a, const SlimCandidate<S, T>& b)
{
    next.row_ids.clear();
    intersection(b.row_ids, a.row_ids, next.row_ids);
    next.pattern.assign(a.pattern);
    next.pattern.insert(b.pattern);
    next.support = count(next.row_ids);
}

template <typename S, typename T>
SlimCandidate<S, T> join(const SlimCandidate<S, T> a, const SlimCandidate<S, T>& b)
{
    auto next = a;
    intersection(b.row_ids, a.row_ids, next.row_ids);
    next.pattern.insert(b.pattern);
    next.support = count(next.row_ids);
    return next;
}

template <typename State>
struct CandidateGeneratorImpl
{
    using state_type = State;

    struct ordering
    {
        constexpr bool operator()(const state_type& a, const state_type& b) const noexcept
        {
            return a.score < b.score;
        }
    };

    struct lexless_pattern
    {
        bool operator()(const state_type x, const state_type& y) const noexcept
        {
            return is_subset(x.pattern, y.pattern);
            // return std::lexicographical_compare(x.pattern.container.begin(),
            //                                     x.pattern.container.end(),
            //                                     y.pattern.container.begin(),
            //                                     y.pattern.container.end());
        }
    };

    struct equals_pattern
    {
        bool operator()(const state_type x, const state_type& y) const noexcept
        {
            return x.support == y.support && sd::equal(y.pattern, x.pattern);
        }
    };

    struct ConstantScoreFunction
    {
        constexpr int operator()(const state_type&) const noexcept { return 1; }
    };

    template <typename Data>
    CandidateGeneratorImpl(const Data& data, size_t min_support, std::optional<size_t> max_depth)
        : max_depth(max_depth), min_support(min_support)
    {
        init(data);
    }

    const state_type& top() const { return candidates.back(); }

    std::optional<state_type> next()
    {
        if (candidates.empty()) return {};

        // std::pop_heap(candidates.begin(), candidates.end(), ordering{});
        auto ret = std::move(candidates.back());
        candidates.pop_back();

        return ret;
    }

    static bool is_candidate_known(const state_type&              x,
                                   size_t                         count_x,
                                   const std::vector<state_type>& candidates)
    {
        return end(candidates) !=
               std::find_if(begin(candidates), end(candidates), [&](const auto& y) {
                   return x.support == y.support && y.pattern.count() == count_x &&
                          equal(y.pattern, x.pattern);
               });
    }

    template <typename score_fn = ConstantScoreFunction>
    size_t combine_two(state_type&       joined,
                       size_t            count_next,
                       const state_type& next,
                       const state_type& other,
                       score_fn&&        score = {})
    {
        if (is_subset(other.pattern, next.pattern)) return 0;

        join(joined, next, other);

        if (joined.support < min_support) return 0;

        auto n = count(joined.pattern);

        if (n <= count_next) return 0;

        if (max_depth && n > *max_depth) return 0;

        joined.score = score(joined);

        return n;
    }

    template<typename score_fn>
    void combine_pairs_allocate_tmp(const state_type& next, score_fn&& score = {})
    {
        novel.resize(singletons.size());

        auto count_next = count(next.pattern);

        auto update_candidate = [&](const auto& i) {
            novel[i].score = 0;
            combine_two(novel[i], count_next, next, singletons[i], score);
        };

#if HAS_EXECUTION_POLICIES
        std::for_each(std::execution::par,
                      tbb::counting_iterator<size_t>(0),
                      tbb::counting_iterator<size_t>(singletons.size()),
                      update_candidate);
#else
#pragma omp parallel for
        for (size_t i = 0; i < singletons.size(); ++i) { update_candidate(i); }
#endif
        candidates.reserve(candidates.size() + novel.size());
        std::copy_if(novel.begin(),
                     novel.end(),
                     std::back_inserter(candidates),
                     [](const auto& x) { return x.score > 0; });
    }

    template <typename score_fn = ConstantScoreFunction>
    void combine_pairs(const state_type& next, score_fn&& score = {})
    {
        combine_pairs_allocate_tmp(next, std::forward<score_fn>(score));
    }

    //     void remove_duplicates()
    //     {

    // #if HAS_EXECUTION_POLICIES
    //         if (candidates.size() > 1024)
    //         {
    //             std::sort(std::execution::par_unseq,
    //                       candidates.begin(),
    //                       candidates.end(),
    //                       lexless_pattern{});

    //             auto ptr = std::unique(std::execution::par_unseq,
    //                                    candidates.begin(),
    //                                    candidates.end(),
    //                                    equals_pattern{});
    //             candidates.erase(ptr, candidates.end());
    //             return;
    //         }
    // #endif
    //         std::sort(candidates.begin(), candidates.end(), lexless_pattern{});
    //         auto ptr = std::unique(candidates.begin(), candidates.end(), equals_pattern{});
    //         candidates.erase(ptr, candidates.end());
    //     }

    void order_candidates()
    {
#if HAS_EXECUTION_POLICIES
        if (candidates.size() > 1024)
        {
            std::sort(
                std::execution::par_unseq, candidates.begin(), candidates.end(), ordering{});
            return;
        }
#endif
        std::sort(candidates.begin(), candidates.end(), ordering{});
    }

    template <typename Fn>
    void prune(Fn&& fn)
    {
#if HAS_EXECUTION_POLICIES
        if (candidates.size() > 1024 * 2)
        {
            auto ptr = std::remove_if(std::execution::par_unseq,
                                      candidates.begin(),
                                      candidates.end(),
                                      std::forward<Fn>(fn));
            candidates.erase(ptr, candidates.end());
            return;
        }
#endif
        auto ptr = std::remove_if(candidates.begin(), candidates.end(), std::forward<Fn>(fn));
        candidates.erase(ptr, candidates.end());
    }

    template <typename score_fn>
    void compute_scores(score_fn&& score)
    {
#if HAS_EXECUTION_POLICIES
        std::for_each(std::execution::par_unseq,
                      std::begin(candidates),
                      std::end(candidates),
                      [&](auto& x) { x.score = score(x); });

#else
#pragma omp parallel for
        for (size_t i = 0; i < candidates.size(); ++i)
        {
            candidates[i].score = score(candidates[i]);
        }
#endif
    }

    template <typename score_fn>
    void compute_scores(const state_type& joined, score_fn&& score)
    {
#if HAS_EXECUTION_POLICIES
        std::for_each(std::execution::par_unseq,
                      std::begin(candidates),
                      std::end(candidates),
                      [&](auto& x) {
                          if (intersects(joined.pattern, x.pattern)) x.score = score(x);
                      });
#else
#pragma omp parallel for
        for (size_t i = 0; i < candidates.size(); ++i)
        {
            if (intersects(joined.pattern, candidates[i].pattern))
                candidates[i].score = score(candidates[i]);
        }
#endif
    }

    template <typename score_fn>
    void expand_from(const state_type& next, score_fn&& score)
    {
        // compute_scores(std::forward<score_fn>(score));
        compute_scores(next, score);
        combine_pairs(next, score);
        order_candidates();
    }

    template <typename score_fn, typename prune_fn>
    void expand_bfs(score_fn&& score, prune_fn&& prune_pred, size_t max_layer_expansion)
    {
        if (!has_next()) return;

        for (size_t layer = 0; layer < max_layer_expansion; ++layer)
        {
            auto curr = candidates.back(); // copy is intentional

            combine_pairs(curr, score);
            this->prune(prune_pred);

            if (!has_next()) break;

            order_candidates();

            if (curr.score >= top().score || sd::equal(curr.pattern, top().pattern)) { break; }
        }
    }

    template <typename score_fn, typename prune_fn>
    void expand_from(const state_type& next,
                     score_fn&&        score,
                     prune_fn&&        prune_pred,
                     size_t            max_search_depth)
    {
        expand_from(next, score);

        if (max_search_depth > 1)
        {
            expand_bfs(score, std::forward<prune_fn>(prune_pred), max_search_depth - 1);
        }
        else
        {
            prune(std::forward<prune_fn>(prune_pred));
        }

        if (has_next() && equal(next.pattern, top().pattern))
        {
            candidates.clear(); // done
        }
    }

    bool   has_next() const { return !candidates.empty(); }
    size_t size() const { return candidates.size(); }

    template <typename Data>
    void init_singletons(Data const& data)
    {
        singletons = std::vector<state_type>(data.dim);

        if (data.size() == 0 || data.dim == 0) return;

        size_t i = 0;
        for (auto& s : singletons)
        {
            s.row_ids.reserve(data.size());
            s.pattern.insert(i++);
        }

        size_t row_index = 0;
        for (const auto& x : data)
        {
            foreach (point(x), [&](size_t i) { singletons[i].row_ids.insert(row_index); })
                ;
            ++row_index;
        }

        for (auto& s : singletons) { s.support = count(s.row_ids); }

        singletons.erase(std::remove_if(singletons.begin(),
                                        singletons.end(),
                                        [&](const auto& s) { return s.support < min_support; }),
                         singletons.end());
    }

    template <typename score_fn = ConstantScoreFunction>
    void initialize_pairs(score_fn&& score)
    {
        size_t     n = singletons.size();
        novel.resize((size_t)std::ceil(n * double(n + 1) / 2));
        auto sym_index = [n](auto i, auto j) {
            if (j < i) { std::swap(i, j); }
            return i * n - (i - 1) * i / 2 + j - i;
        };

#if HAS_EXECUTION_POLICIES
        std::for_each(
            std::execution::par,
            tbb::counting_iterator<size_t>(0),
            tbb::counting_iterator<size_t>(n),
            [&](size_t i) {
                for (size_t j = i + 1; j < n; ++j)
                {
                    combine_two(novel[sym_index(i, j)], 1, singletons[i], singletons[j], score);
                }
            });
#else
        state_type joined;
#pragma omp parallel for schedule(dynamic, 1)
        for (size_t i = 0; i < singletons.size(); ++i)
        {
            for (size_t j = i + 1; j < singletons.size(); ++j)
            {
                combine_two(novel[sym_index(i, j)], 1, singletons[i], singletons[j], score); 
            }
        }
#endif
        candidates.reserve(candidates.size() + novel.size());
        std::copy_if(std::make_move_iterator(novel.begin()),
                     std::make_move_iterator(novel.end()),
                     std::back_inserter(candidates),
                     [](const auto& x) { return x.score > 0; });
        novel.clear();
        order_candidates();
    }

    template <typename Data>
    void init(Data const& data)
    {
        init_singletons(data);
        candidates.clear();
        candidates.reserve(data.dim * 3);
    }

    template <typename Patternset, typename score_fn = ConstantScoreFunction>
    void initialize_expansion(const Patternset& patternset, score_fn&& score = {})
    {
        for (const auto& x : patternset) { combine_pairs(point(x), score); }
        order_candidates();
    }

private:
    std::optional<size_t>   max_depth;
    size_t                  min_support = 2;
    std::vector<state_type> singletons;
    std::vector<state_type> candidates;

    std::vector<state_type> novel;

    static_assert(std::is_swappable_v<state_type>);
};

template <typename S, typename T>
using CandidateGenerator = CandidateGeneratorImpl<SlimCandidate<S, T>>;

} // namespace disc
} // namespace sd
