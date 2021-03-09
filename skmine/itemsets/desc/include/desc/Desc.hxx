#pragma once

#include "desc/PatternAssignment.hxx"
#include <desc/Component.hxx>
#include <desc/Composition.hxx>
#include <desc/PatternsetMiner.hxx>
#include <desc/utilities/ModelPruning.hxx>

namespace sd::disc
{
struct IDesc : DefaultPatternsetMinerInterface, DefaultAssignment
{
    template <typename C, typename Config>
    static auto finish(C& c, const Config& cfg)
    {
        prune_model(c, cfg);
    }
};
} // namespace sd::disc

// #include <chrono>
// template <typename T = std::chrono::milliseconds, typename Fn>
// auto timeit(Fn&& fn)
// {
//     namespace ch = std::chrono;
//     using clk = std::chrono::high_resolution_clock;
//     using namespace std::chrono_literals;

//     auto before = clk::now();
//     std::forward<Fn>(fn)();
//     return (clk::now() - before) / 1ms;
//     // return ch::duration_cast<T>(clk::now() - before);
// }

// namespace sd::disc {

// template<typename Trait>
// class Explanation
// {
//     void contrast() {}
//     void shared() {}
//     void characteristic() {}
//     void emerging() {}
//     void confidence() {}
// };

// template<typename Trait, typename MinerType = DefaultPatternsetMinerInterface>
// auto desc(Dataset<typename Trait::pattern_type> x, Config cfg, MinerType && m = {})
// {
//     Component<Trait> c;
//     c.data  = std::move(x);
//     elapsed_time_fit = timeit([&] { discover_patterns_generic(c, cfg,
//     std::forward<MinerType>(m)); }); return c;
// }

// template<typename Trait, typename MinerType = DefaultPatternsetMinerInterface>
// auto desc(Dataset<typename Trait::pattern_type> x, Config cfg,
// std::optional<std::vector<size_t>> const& y = std::nullopt, MinerType && m = {})
// {
//     using Pi = PartitionedDataset<typename Trait::pattern_type>;
//     Pi pi = (y && y->size() > 0) ? Pi{std::move(x), *y} : Pi{std::move(x)};
//     return desc<Trait>(std::move(pi), cfg, std::forward<MinerType>(m));
// }

// template<typename Trait, typename MinerType = DefaultPatternsetMinerInterface>
// auto desc(PartitionedDataset<typename Trait::pattern_type> x, Config cfg, MinerType && m =
// {})
// {
//     Composition<Trait> c;
//     c.data = std::move(x);
//     initialize_model(c, *this);

//     auto initial_encoding = c.encoding = encode(c, *this);

//     elapsed_time_fit = timeit([&] { discover_patterns_generic(c, *this, MinerType{}); });

//     c.initial_encoding = initial_encoding;
//     c.data.revert_order();

//     return c;
// }

// template<typename Trait, typename MinerType = DefaultPatternsetMinerInterface>
// class OneClassDESC
// {
//     using trait_type = Trait;

//     Config cfg;
//     Component<trait_type> state;

// public:
//     using pattern_type = typename trait_type::pattern_type;
//     using float_type = typename trait_type::float_type;
//     std::chrono::milliseconds elapsed_time_fit;
//     void fit(Dataset<pattern_type> x)
//     {
//         Component<trait_type> c;
//         c.data  = std::move(x);
//         elapsed_time_fit = timeit([&] { discover_patterns_generic(c, cfg); });
//         this->state = std::move(c);
//     }
// };

// template<typename Trait, typename MinerType = DefaultPatternsetMinerInterface>
// class DESC :  public sd::disc::Config
// {
//     using trait_type = Trait;

//     std::variant<Composition<trait_type>, Component<trait_type>> state;

// public:
//     using pattern_type = typename trait_type::pattern_type;
//     using float_type = typename trait_type::float_type;

//     std::chrono::milliseconds elapsed_time_fit;

//     void fit(Dataset<pattern_type> x, std::optional<std::vector<size_t>> const& y =
//     std::nullopt)
//     {
//         if (y && y->size() != 0)
//         {
//             Composition<trait_type> c;
//             c.data = PartitionedData<pattern_type>(std::move(x), *y);
//             initialize_model(c, *this);
//             auto initial_encoding = c.encoding = encode(c, *this);

//             // auto before = clk::now();
//             elapsed_time_fit = timeit([&] { discover_patterns_generic(c, *this, MinerType{});
//             });
//             // elapsed_time_fit = (clk::now() - before) / 1ms;

//             c.initial_encoding                 = initial_encoding;
//             c.data.revert_order();

//             this->state = std::move(c);
//         }
//         else
//         {
//             Component<trait_type> c;
//             c.data  = std::move(x);
//             elapsed_time_fit = timeit([&] { discover_patterns_generic(c, *this, MinerType{});
//             });

//             this->state = std::move(c);
//         }
//     }

//     void characterize(PartitionedData<pattern_type> x, Dataset<pattern_type> const & summary)
//     {
//         Composition<trait_type> c;
//         c.data = std::move(x);
//         insert_missing_singletons(c);
//         for(const auto& [x] : summary)
//         {
//             c.summary.insert(frequency<float_type>(x, c.data), x);
//         }
//         initialize_model(c, *this);
//         characterize_components(c, *this);
//         this->state = std::move(c);
//     }

//     void characterize(Dataset<pattern_type> x, const Dataset<pattern_type>& summary,
//     std::optional<std::vector<size_t>> const& y = std::nullopt)
//     {
//         if (y && y->size() != 0)
//         {
//             characterize(PartitionedData<pattern_type>{std::move(x), *y}, summary);
//         }
//         else
//         {
//             Component<trait_type> c;
//             c.data = std::move(x);
//             insert_missing_singletons(c);
//             for(const auto& [x] : summary)
//             {
//                 c.summary.insert(frequency<float_type>(x, c.data), x);
//             }
//             initialize_model(c, *this);

//             this->state = std::move(c);
//         }
//     }
//     ndarray<float_type, 2> confidence_matrix() const {}

//     auto log_likelihood(PartitionedData<pattern_type> const& xs)
//     {
//         if (std::holds_alternative<Component<trait_type>>(state))
//         {
//             const auto& c = std::get<Component<trait_type>>(state);
//             return std::reduce(std::execution::par_unseq, begin(x), end(x), float_type{},
//             [&](auto l, const auto& xi)
//             {
//                 return l - c.model.log_probability(point(xi));
//             });
//         }
//         else
//         {
//             const auto& c = std::get<Composition<trait_type>>(state);
//             return std::reduce(std::execution::par_unseq, begin(x), end(x), float_type{},
//             [&](auto l, const auto& xi)
//             {
//                 return l - c.models[label(xi)].log_probability(point(xi));
//             });
//         }
//     }

//     auto log_likelihood(Dataset<pattern_type> x, std::optional<std::vector<size_t>> const& y
//     = std::nullopt)
//     {
//         if (std::holds_alternative<Component<trait_type>>(state))
//         {
//             const auto& c = std::get<Component<trait_type>>(state);
//             return std::reduce(std::execution::par_unseq, begin(x), end(x), float_type{},
//             [&](auto l, const auto& xi)
//             {
//                 return l - c.model.log_probability(point(xi));
//             });
//         }
//         else if(y && y.size() > 0)
//         {
//             const auto& c = std::get<Composition<trait_type>>(state);

//             auto r = zip(x->template col<0>(), *y);

//             return std::reduce(std::execution::par_unseq, r.begin(), r.end(), float_type{},
//             [&](auto l, const auto& xi)
//             {
//                 return l - c.models[label(xi)].log_probability(point(xi));
//             });
//         }
//         else
//         {
//             const auto& c = std::get<Composition<trait_type>>(state);

//             if (c.models.size() != 1)
//             {
//                 throw std::runtime_error{"cannot encode multiple classes with this model"};
//             }

//             return std::reduce(std::execution::par_unseq, begin(x), end(x), float_type{},
//             [&](auto l, const auto& xi)
//             {
//                 return l - c.models.first().log_probability(point(xi));
//             });
//         }

//     }

//     Explanation<trait_type> explanation() const
//     {
//     }

//     sd::ndarray<size_t,2> predict_probabilities(const Dataset<pattern_type>& x) const
//     {}

//     std::vector<size_t> predict(const Dataset<pattern_type>& x) const
//     {
//         bool is_empty = false;
//         std::visit([&](auto&& s){ is_empty = s.data.size() == 0; }, state);
//         if (is_empty) return {};

//         std::vector<size_t> y(x.size(), 0);

//         if (std::holds_alternative<Component<trait_type>>(state))
//         {
//             return y;
//         }

//         auto& c = std::get<Composition<trait_type>>(state);
//         const auto k = c.data.num_components();

//         for(size_t i = 0; i < y.size(); ++i)
//         {
//             std::pair<size_t, float_type> y_i = {};
//             for(size_t j = 0; j < k; ++j)
//             {
//                 auto p = c.models[j].expectation(x.point(i));
//                 if(y_i.second < p)
//                 {
//                     y_i = {j, p};
//                 }
//                 y[i] = y_i.first;
//             }
//         }

//         return y;
//     }
// };

// }
