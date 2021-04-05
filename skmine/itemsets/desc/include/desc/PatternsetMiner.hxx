#pragma once

#include <desc/CandidateGeneration.hxx>
#include <desc/DescHeuristic.hxx>
#include <desc/PatternAssignment.hxx>
#include <desc/utilities/EmptyCallback.hxx>

namespace sd::disc
{

template <typename Trait, typename Candidate>
bool is_allowed(const Component<Trait>& c, const Candidate& x)
{
    return c.model.is_allowed(x.pattern);
}

template <typename Trait, typename Candidate>
bool is_allowed(const Composition<Trait>& c, const Candidate& x)
{
    return std::any_of(
        begin(c.models), end(c.models), [&](const auto& m) { return m.is_allowed(x.pattern); });
}

template <typename Trait, typename Config>
void prepare(Component<Trait>& c, const Config& cfg)
{
    if (c.summary.empty() || c.model.model.dim != c.data.dim) initialize_model(c, cfg);
}

template <typename Trait, typename Config>
void prepare(Composition<Trait>& c, const Config& cfg)
{
    if (!check_invariant(c)) { initialize_model(c, cfg); }

    c.masks = construct_component_masks(c);
}

struct DefaultPatternsetMinerInterface
{
    template <typename C, typename Candidate, typename Config>
    static auto heuristic(C& c, Candidate& x, const Config&)
    {
        return sd::disc::desc_heuristic(c, x);
    }
    template <typename C, typename Candidate, typename Config>
    static auto is_allowed(C& c, const Candidate& x, const Config&)
    {
        return sd::disc::is_allowed(c, x);
    }
    template <typename C, typename Candidate, typename Config>
    static auto insert_into_model(C& c, Candidate& x, const Config& cfg)
    {
        return sd::disc::find_assignment(c, x, cfg);
    }
    template <typename C, typename Config>
    static void prepare(C& c, const Config& cfg)
    {
        sd::disc::prepare(c, cfg);
    }
    template <typename C, typename Config>
    static void finish(C&, const Config&)
    {
    }
};

template <typename C,
          typename I    = DefaultPatternsetMinerInterface,
          typename Info = EmptyCallback>
void discover_patterns_generic(C& s, const Config& cfg, I fn = {}, Info&& info = {})
{
    using patter_type = typename C::pattern_type;
    using float_type  = typename C::float_type;
    using generator   = CandidateGenerator<patter_type, float_type>;
    using clk         = std::chrono::high_resolution_clock;

    assert(s.data.dim != 0);

    fn.prepare(s, cfg);

    info(std::as_const(s));

    auto score_fn = [&](auto& x) { return fn.heuristic(s, x, cfg); };
    auto prune_fn = [&](auto& x) { return x.score <= 0 || !fn.is_allowed(s, x, cfg); };

    auto max_depth = cfg.max_pattern_size.value_or(cfg.max_factor_width);
    auto gen       = generator(s.data, cfg.min_support, max_depth);

    gen.create_pair_candidates(score_fn);

    if (cfg.search_depth > 1) { 
        gen.expand_bfs(score_fn, prune_fn, cfg.search_depth); 
    }

    size_t     items_used = 0;
    size_t     patience   = cfg.max_patience;
    const auto start_time = clk::now();

    for (size_t it = 0; it < cfg.max_iteration && gen.has_next(); ++it)
    {
        auto c = gen.next();

        if (c && fn.insert_into_model(s, *c, cfg))
        {
            patience   = std::min(patience * 2, cfg.max_patience);
            items_used = items_used + 1;

            info(std::as_const(s));
        }
        else if (patience-- == 0)
            break;

        if (cfg.max_time && (clk::now() - start_time) > *cfg.max_time) { break; }

        if (cfg.max_patternset_size && items_used >= *cfg.max_patternset_size) { break; }

        if (c) { gen.expand_from(*c, score_fn, prune_fn, cfg.search_depth); }
    }

    fn.finish(s, cfg);
}

} // namespace sd::disc