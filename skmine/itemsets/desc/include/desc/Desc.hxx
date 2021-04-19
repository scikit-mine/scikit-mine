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
        prune_composition(c, cfg, DefaultAssignment{});
    }
};
} // namespace sd::disc
