// #pragma once

// #include <desc/distribution/StaticFactorModel.hxx>
// #include <setcover/GreedySetcover.hxx>

// namespace sd::disc
// {

// struct SetcoverWeightFunction
// {
//     template <typename Set, typename To_Cover>
//     float operator()(size_t, const Set& f, const To_Cover& y) const noexcept
//     {
//         const auto s = size_of_intersection(y, f.range);

//         if (s > 1 && f.factor.itemsets.get_precomputed_expectation(y))
//         {
//             return std::numeric_limits<float>::max();
//         }
//         const auto t = f.factor.itemsets.width();
//         return s == t ? (static_cast<float>(s)) : -1.f;
//     }
// };

// struct SuperSetcoverWeightFunction
// {
//     template <typename Set, typename To_Cover>
//     float operator()(size_t, const Set& f, const To_Cover& y) const noexcept
//     {
//         const auto s = size_of_intersection(y, f.range);

//         if (s > 1 && f.factor.itemsets.get_precomputed_expectation(y))
//         {
//             return std::numeric_limits<float>::max();
//         }

//         return static_cast<float>(s);
//     }
// };

// struct WeightedSuperSetcoverWeightFunction
// {
//     template <typename Set, typename To_Cover>
//     float operator()(size_t, const Set& f, const To_Cover& y) const noexcept
//     {
//         const auto s = size_of_intersection(y, f.range);

//         if (s > 1 && f.factor.itemsets.get_precomputed_expectation(y))
//         {
//             return std::numeric_limits<float>::max();
//         }

//         const auto t = f.factor.itemsets.width();
//         return static_cast<float>(s) / (t - s + 1);
//     }
// };

// struct SetMinusFactor
// {
//     template <typename To_Cover, typename Set>
//     void operator()(To_Cover& s, const Set& f) const noexcept
//     {
//         sd::setminus(s, f.range);
//     }
// };

// template <typename U, typename X, typename Visitor>
// void factorize_patterns_dynamically(const std::vector<Factor<U>>& factors, X& x, Visitor&& f)
// {
//     using W = SuperSetcoverWeightFunction;
//     setcover::greedy_set_cover(
//         factors,
//         x,
//         [&](size_t i) { f(factors[i], i); },
//         {1'000'000, -1},
//         W{},
//         SetMinusFactor{});
// }

// template <typename U, typename X, typename Visitor>
// void factorize_dynamically(const Factorization<U>& phi, const X& x, Visitor&& f)
// {
//     thread_local itemset<typename U::pattern_type> y;
//     y.clear();
//     y.assign(x);

//     factorize_patterns_dynamically(phi.factors, y, [&](const auto& phi_i, size_t i) { f(phi_i, i, false); });
//     foreach (y, [&](size_t i) { f(phi.singleton_factors[i], i, true); });
// }

// template <typename factor_type>
// void erase_uncovered_singletons(factor_type& f, bool estimate)
// {
//     auto&       s   = f.factor.singletons.set;
//     const auto& r   = f.range;
//     const auto  pos = std::remove_if(
//         s.begin(), s.end(), [&r](const auto& i) { return !is_subset(i.element, r); });

//     if (pos != s.end())
//     {
//         s.erase(pos, s.end());

//         if (estimate) estimate_model(f.factor);
//         f.factor.itemsets.num_singletons = f.factor.singletons.size();
//     }
// }

// template <typename factor_type, typename float_type, typename T>
// void create_mindiv_factor(factor_type& next,
//                           float_type   frequency,
//                           const T&     t,
//                           size_t       max_factor_size,
//                           size_t       max_factor_width,
//                           bool         estimate)
// {
//     assert(!is_singleton(t));
//     // singletons + most informative next pattern

//     factor_type replacement(next.factor.singletons.dim);

//     replacement.factor.singletons              = next.factor.singletons;
//     replacement.factor.itemsets.num_singletons = replacement.factor.singletons.size();
//     replacement.range.insert(t);
//     replacement.factor.insert(frequency, t, estimate);

//     auto& xs = next.factor.itemsets.set;

//     thread_local itemset<tag_dense> in_use;
//     in_use.clear();

//     // greedy algorithm to select the sets from next that happen to increase the likelihood
//     for (size_t count = 1, n = xs.size(); count <= max_factor_size;)
//     {
//         std::pair best{n, float_type(0)};
//         for (size_t j = 0; j < n; ++j)
//         {
//             if (in_use.test(j)) { continue; }

//             if (!intersects(xs[j].point, t))
//             {
//                 in_use.insert(j);
//                 continue;
//             }

//             if (size_of_union(xs[j].point, replacement.range) > max_factor_width)
//             {
//                 in_use.insert(j);
//                 continue;
//             }
//             using std::log2;

//             auto p = expectation(replacement.factor, xs[j].point);
//             auto q = xs[j].frequency;
//             auto g = (q * log2(q / p)) + (p * log2(p / q)); // assignment_score

//             if (g >= best.second) { best = {j, g}; }
//         }

//         if (best.first < n && best.second != 0)
//         {
//             ++count;
//             in_use.insert(best.first);
//             const auto& y = xs[best.first].point;
//             const auto& q = xs[best.first].frequency;
//             replacement.range.insert(y);
//             replacement.factor.insert(q, y, estimate);
//         }
//         else
//         {
//             break;
//         }
//     }

//     erase_uncovered_singletons(replacement, estimate);
//     next = std::move(replacement);
// }

// template <typename factor_type, typename T>
// void prune_factor_set_cover_based(factor_type& next,
//                                   const T&     t,
//                                   size_t       max_factor_size,
//                                   bool         estimate)
// {
//     auto&       is = next.factor.itemsets.set;
//     factor_type replacement(next.factor.singletons.dim);

//     auto weight = [](size_t, const auto& xs, const auto& ys) {
//         size_t s = sd::size_of_intersection(xs.point, ys);
//         return s;
//         // return s == count(xs.point) ? s : size_t(0);
//     };
//     auto setminus = [](auto& ys, const auto& xs) { sd::setminus(ys, xs.point); };
//     auto report   = [&](size_t i) {
//         replacement.factor.insert(is[i].frequency, is[i].point);
//         replacement.range.insert(is[i].point);
//     };

//     // setcover::greedy_set_cover_by_value(is, t, report, {max_factor_size - 1}, weight,
//     // setminus);
//     disc::itemset<disc::tag_dense> ignore(is.size());

//     setcover::greedy_set_cover_no_minus(ignore, is, t, report, {max_factor_size - 1}, weight);

//     for (const auto& i : next.factor.singletons.set)
//     {
//         if (is_subset(i.element, replacement.range))
//         {
//             replacement.factor.insert_singleton(i.frequency, i.element);
//         }
//     }
//     if (estimate) estimate_model(replacement.factor);
//     next = std::move(replacement);
// }

// template <typename factor_type, typename T>
// void prune_factor_overlapping_based(factor_type& next,
//                                     const T&     t,
//                                     size_t       max_factor_size,
//                                     bool         estimate)
// {
//     // keep most essential sets
//     auto& is = next.factor.itemsets.set;
//     std::sort(std::begin(is), std::end(is), [&t](const auto& a, const auto& b) {
//         return size_of_intersection(a.point, t) > size_of_intersection(b.point, t);
//     });
//     is.resize(max_factor_size);
//     next.range.clear();
//     for (const auto& i : is) { next.range.insert(i.point); }
//     next.range.insert(t);
//     auto& sset = next.factor.singletons.set;
//     sset.erase(std::remove_if(std::begin(sset),
//                               std::end(sset),
//                               [&](const auto& s) { return !is_subset(s.element, next.range); }),
//                std::end(sset));
//     next.factor.itemsets.num_singletons = sset.size();

//     if (estimate) estimate_model(next.factor);
// }

// template <typename U, typename V, typename Underlying_Factor_Type = MaxEntFactor<U, V>>
// struct RelaxedFactorModel
// {
//     using float_type            = V;
//     using pattern_type          = U;
//     using index_type            = std::size_t;
//     using underlying_model_type = Underlying_Factor_Type;
//     using factor_type           = Factor<underlying_model_type>;

//     explicit RelaxedFactorModel(size_t dim) : dim(dim) { init(dim); }
//     RelaxedFactorModel() = default;

//     size_t dim              = 0;
//     size_t max_factor_size  = 5;
//     size_t max_factor_width = 8;

//     Factorization<Underlying_Factor_Type> phi;

//     size_t num_itemsets() const { return phi.factors.size(); }
//     size_t dimension() const { return dim; }
//     size_t size() const { return num_itemsets() + dimension(); }

//     void init(size_t d)
//     {
//         dim = d;
//         clear();
//         init_singletons(phi.singleton_factors, dim);
//     }

//     void clear()
//     {
//         phi.factors.clear();
//         phi.singleton_factors.clear();
//     }

//     void insert_singleton(float_type frequency, const index_type element, bool estimate)
//     {
//         set_singleton(phi.singleton_factors, frequency, element, estimate);
//         // disc::insert_singleton(phi.singleton_factors, dim, frequency, element, estimate);
//     }

//     template <typename T>
//     void insert_singleton(float_type frequency, const T& t, bool estimate)
//     {
//         insert_singleton(frequency, static_cast<index_type>(front(t)), estimate);
//     }

//     template <typename T>
//     void insert_pattern_join_only_singletons(float_type frequency, const T& t, bool estimate)
//     {
//         factor_type next(this->dim);
//         foreach (t, [&](size_t i) { join_factors(next, phi.singleton_factors[i]); })
//             ;

//         next.range.insert(t);
//         next.factor.insert(frequency, t, estimate);
//         phi.factors.emplace_back(std::move(next));
//     }

//     template <typename T>
//     void insert_pattern(float_type fr,
//                         const T&   t,
//                         size_t     max_factor_size,
//                         size_t     max_factor_width,
//                         bool       estimate)
//     {

//         if (max_factor_size <= 1)
//         {
//             insert_pattern_join_only_singletons(fr, t, estimate);
//             return;
//         }

//         // itemset<U>  x_buf(t);
//         factor_type next(this->dim);

//         // factorize_dynamically(
//         //     *this, this->dim, x_buf, [&](auto& f, size_t, bool) { join_factors(next, f); });

//         factorize_dynamically(
//             phi, t, [&](const auto& f, size_t, bool) { join_factors(next, f); });

//         const auto next_elems = next.factor.itemsets.set.size();
//         const auto next_items = next.factor.singletons.set.size();

//         if ((next_elems >= max_factor_size || next_items > max_factor_width))
//         {
//             create_mindiv_factor(next, fr, t, max_factor_size, max_factor_width, true);
//         }
//         else
//         {
//             next.factor.insert(fr, t, estimate);
//             next.range.insert(t);
//         }

//         this->phi.factors.emplace_back(std::move(next));
//     }

//     template <typename T>
//     void insert_pattern(float_type frequency, const T& t, bool estimate)
//     {
//         insert_pattern(frequency, t, max_factor_size, max_factor_width, estimate);
//     }

//     template <typename T>
//     void insert(float_type frequency, const T& t, bool estimate)
//     {
//         if (is_singleton(t))
//             insert_singleton(frequency, t, estimate);
//         else
//             insert_pattern(frequency, t, estimate);
//     }

//     template <typename T>
//     bool is_allowed(const T&, size_t, size_t) const
//     {
//         return true;
//     }

//     template <typename T>
//     bool is_allowed(const T& t) const
//     {
//         return is_allowed(t, max_factor_size, max_factor_width);
//     }
// };
// template <typename S, typename T, typename U, typename X, typename Visitor>
// void factorize(const RelaxedFactorModel<S, T, U>& pr, const X& x, Visitor&& f)
// {
//     factorize_dynamically(pr.phi, x, std::forward<Visitor>(f));
// }
// template <typename U, typename V, typename X, typename Visitor>
// void factorize_patterns(const RelaxedFactorModel<U, V>& pr, const X& x, Visitor&& f)
// {
//     thread_local itemset<typename RelaxedFactorModel<U, V>::pattern_type> y;
//     y.clear();
//     y.assign(x);

//     factorize_patterns_dynamically(pr.phi.factors, y, f);
// }
// template <typename S, typename T, typename U, typename F = double>
// void estimate_model(RelaxedFactorModel<S, T, U>&       m,
//                     IterativeScalingSettings<F> const& opts = {})
// {
//     estimate_model(m.phi.factors, opts);
// }
// template <typename S, typename T, typename U, typename X, typename Visitor>
// void factorize_singletons(const RelaxedFactorModel<S, T, U>& pr, const X& x, Visitor&& v)
// {
//     factorize_singletons(pr.phi, x, std::forward<Visitor>(v));
// }
// } // namespace sd::disc