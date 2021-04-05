#pragma once

#include <desc/PatternAssignment.hxx>
#include <desc/Component.hxx>
#include <desc/Composition.hxx>
#include <desc/PatternsetMiner.hxx>
#include <desc/utilities/ModelPruning.hxx>
// #include <relent/ReaperModelPruning.hxx>

namespace sd::disc
{

struct IDesc : DefaultPatternsetMinerInterface, DefaultAssignment
{
    template <typename C, typename Config>
    static auto finish(C& c, const Config& cfg)
    {
        // using dist_t = typename std::decay_t<C>::distribution_type;
        // if constexpr (is_dynamic_factor_model<dist_t>())
        // {
        //     reaper_prune_pattern_composition(c, cfg);
        // }
        // else
        // {
            prune_pattern_composition(c, cfg);
        // }
    }
};
} // namespace sd::disc
