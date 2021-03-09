#pragma once

#include <chrono>
#include <optional>

namespace sd
{
namespace disc
{

struct Config
{
    double alpha            = 0.01;
    size_t min_support      = 2;
    size_t search_depth     = 10;
    size_t max_patience     = 20;
    size_t max_factor_width = 15;
    size_t max_factor_size  = 8;
    size_t max_iteration    = std::numeric_limits<size_t>::max();

    std::optional<size_t>                    max_pattern_size;
    std::optional<size_t>                    max_patternset_size;
    std::optional<std::chrono::milliseconds> max_time;
};

} // namespace disc
} // namespace sd