#pragma once

#include <desc/Component.hxx>
#include <desc/Composition.hxx>
#include <desc/Settings.hxx>

namespace sd
{
namespace disc
{

struct DefaultAssignment
{
    template <typename Trait, typename Pattern>
    static auto confidence(const Composition<Trait>&         c,
                           size_t                            index,
                           const typename Trait::float_type& q,
                           const Pattern&                    x,
                           const Config&)
    {
        using std::log2;
        return q > 0 && log2(q / c.models[index].expectation(x)) > 0;
    }

    template <typename Trait, typename Pattern>
    static auto confidence(const Component<Trait>&           c,
                           const typename Trait::float_type& q,
                           const Pattern&                    x,
                           const Config&)
    {
        using std::log2;
        return q > 0 && log2(q / c.model.expectation(x)) > 0;
    }
};

template <typename Trait, typename Candidate>
void insert_pattern_to_summary(Composition<Trait>& c, const Candidate& x)
{
    using float_type = typename Trait::float_type;

    assert(c.frequency.extent(1) > 0);

    c.summary.insert(x.pattern);

    auto row = c.frequency.extent(0);
    c.frequency.resize(sd::layout<2>({row + 1, c.data.num_components()}));

    auto new_q = c.frequency[c.frequency.extent(0) - 1];
    for (size_t j = 0; j < c.data.num_components(); ++j)
    {
        auto s   = size_of_intersection(x.row_ids, c.masks[j]);
        new_q(j) = static_cast<float_type>(s) / c.data.subset(j).size();
    }
}

template <typename Trait, typename Candidate, typename Interface = DefaultAssignment>
bool find_assignment_impl(Composition<Trait>& c,
                          const Candidate&    x,
                          const Config&       cfg,
                          const Interface&    f = {})
{
    using float_type = typename Trait::float_type;
    if (x.score <= 0)
        return false;

    thread_local std::vector<float_type> conf; 
    conf.resize(c.data.num_components()); 

    size_t counter = 0;
    for (size_t i = 0; i < c.data.num_components(); ++i)
    {
        auto s  = size_of_intersection(x.row_ids, c.masks[i]);
        auto q = static_cast<float_type>(s) / c.data.subset(i).size();
        auto& pr = c.models[i];
        conf[i] = f.confidence(c, i, q, x.pattern, cfg);
        if (pr.is_allowed(x.pattern) && conf[i] > 0)
        {
            pr.insert(q, x.pattern, true);
            c.assignment[i].insert(c.summary.size());
            // c.confidence(c.summary.size(), i) = conf;
            // c.frequency(c.summary.size(), i) = q;
            ++counter;
        }
    }

    if (counter)
    {
        // c.confidence.push_back();
        auto row = c.confidence.length();
        c.confidence.resize(sd::layout<2>({row + 1, c.data.num_components()}));
        for (size_t i = 0; i < conf.size(); ++i)
        {
            c.confidence(row, i) = conf[i];
        }
        insert_pattern_to_summary(c, x);
        
        return true;
    }
    else
    {
        return false;
    }
}

template <typename Trait, typename Candidate>
bool find_assignment_impl_first(Composition<Trait>& c, const Candidate& x, const Config&)
{
    using float_type = typename Trait::float_type;

    if (x.score <= 0 || !c.models[0].is_allowed(x.pattern))
        return false;

    auto q = static_cast<float_type>(x.support) / c.data.size();

    c.models[0].insert(q, x.pattern, true);
    c.assignment[0].insert(c.summary.size());
    c.confidence.push_back(x.score);
    insert_pattern_to_summary(c, x);
    return true;
}

template <typename Trait, typename Candidate, typename Interface = DefaultAssignment>
bool find_assignment(Composition<Trait>& c,
                     const Candidate&    x,
                     const Config&       cfg,
                     const Interface&    f = {})
{
    if (c.data.num_components() == 1)
    {
        return find_assignment_impl_first(c, x, cfg);
    }
    else
    {
        return find_assignment_impl(c, x, cfg, f);
    }
}

template <typename Trait, typename Candidate, typename Interface = DefaultAssignment>
bool find_assignment(Component<Trait>& c,
                     const Candidate&  x,
                     const Config&,
                     const Interface& = {})
{
    using float_type = typename Trait::float_type;

    if (x.score <= 0)
        return false;

    auto q = static_cast<float_type>(x.support) / c.data.size();

    c.model.insert(q, x.pattern, true);
    c.summary.insert(x.pattern);
    c.frequency.push_back(q);
    c.confidence.push_back(x.score);

    return true;
}

} // namespace disc
} // namespace sd