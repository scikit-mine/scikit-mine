#pragma once

#include "desc/storage/Itemset.hxx"
#include <desc/utilities/BoostMultiprecision.hxx>

#include <algorithm>
#include <chrono>
#include <exception>
#include <optional>

#include <desc/CharacterizeComponents.hxx>
#include <desc/Component.hxx>
#include <desc/Composition.hxx>
#include <desc/Desc.hxx>
#include <desc/DescMDL.hxx>
#include <desc/utilities/BiMap.hxx>

#include <stdexcept>
#include <type_traits>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

namespace sd::disc
{

template <typename T = std::chrono::milliseconds, typename Fn>
auto timeit(Fn&& fn)
{
    namespace ch = std::chrono;
    // using namespace std::chrono_literals;

    auto before = ch::high_resolution_clock::now();
    std::forward<Fn>(fn)();
    auto after = ch::high_resolution_clock::now();
    // return (after - before) / 1ms;
    return ch::duration_cast<T>(after - before);
}

using real_matrix  = sd::ndarray<double, 2>;
using label_vector = std::vector<int>;

template <typename S, typename T>
auto to_dataset(Dataset<T> const& xss)
    -> std::conditional_t<std::is_same_v<S, T>, Dataset<S> const&, Dataset<S>>
{
    if constexpr (std::is_same_v<S, T>) { return xss; } // std::forward<Dataset<S>>(xss); }
    else
    {
        itemset<S> buffer;
        Dataset<S> out;
        out.reserve(xss.size());
        for (const auto& [xs] : xss)
        {
            buffer.clear();
            sd::foreach (xs, [&](size_t i) { buffer.insert(i); });
            out.insert(buffer);
        }
        return out;
    }
}

struct Explaination
{
public:
    AssignmentMatrix                 A;
    std::vector<std::vector<size_t>> set;
    std::chrono::milliseconds        elapsed;
    Config                           cfg;
    real_matrix                      confidence_matrix;

    virtual const std::vector<std::vector<size_t>>& patterns() const { return set; }

    virtual std::vector<size_t> contrasting(const std::vector<size_t>& I,
                                            const std::vector<size_t>& J) const
    {
        itemset<tag_sparse> a, b;
        for (auto i : I) a.insert(A.at(i));
        for (auto j : J) b.insert(A.at(j));

        std::vector<size_t> c;
        std::set_symmetric_difference(
            a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(c));
        return c;
    }
    virtual std::vector<size_t> emerging(const std::vector<size_t>& I,
                                         const std::vector<size_t>& J) const
    {
        itemset<tag_sparse> a, b;
        for (auto i : I) a.insert(A.at(i));
        for (auto j : J) setminus(a, A.at(j));
        return {a.container.begin(), a.container.end()};
    }
    virtual std::vector<size_t> characteristic(size_t i) const { return A.at(i).container; }
    virtual std::vector<size_t> common(const std::vector<size_t>& I) const
    {
        if (I.empty()) return {};

        itemset<tag_sparse> a;

        a.insert(A.at(I.front()));
        for (auto i : I) intersection(A.at(i), a);
        return {a.container.begin(), a.container.end()};
    }
    virtual std::vector<size_t> unique(const std::vector<size_t>& I) const
    {
        itemset<tag_sparse> a;
        for (auto i : I) a.insert(A.at(i));
        for (size_t i = 0; i < A.size(); ++i)
        {
            if (std::find(I.begin(), I.end(), i) == I.end()) { setminus(a, A.at(i)); }
        }
        return {a.container.begin(), a.container.end()};
    }
    // std::vector<std::vector<size_t>> patterns(const std::vector<size_t>& I) const
    // {
    //     std::vector<std::vector<size_t>> result(I.size());
    //     for (auto i : I)
    //     {
    //         if (!set.at(i).empty()) result.push_back(set[i]);
    //     };
    //     return result;
    // }

    template <typename T>
    void extract_explanation(Composition<T> const& c)
    {
        A.resize(c.data.num_components());
        for (auto& a : A) a.clear();

        if (c.summary.size() <= c.data.dim) { return; }

        set.reserve(c.summary.size() - c.data.dim);

        for (size_t i = c.data.dim; i < c.summary.size(); ++i)
        {
            if (count(c.summary.point(i)) > 0)
            {
                auto& next = set.emplace_back();
                next.reserve(count(c.summary.point(i)));
                foreach (c.summary.point(i), [&](size_t index) { next.push_back(index); })
                    ;
            }
        }

        for (size_t j = 0; j < c.data.num_components(); ++j)
        {
            const auto& aj = c.assignment[j];
            for (size_t i = 0; i < aj.container.size(); ++i)
            {
                if (aj.container[i] >= c.data.dim) A.at(j).insert(aj.container[i] - c.data.dim);
            }
        }

        this->confidence_matrix.resize(c.confidence.extent(0) - c.data.dim,
                                       c.confidence.extent(1));
        for (size_t i = c.data.dim; i < c.confidence.extent(0); ++i)
            for (size_t j = 0; j < c.confidence.extent(1); ++j)
            {
                confidence_matrix(i - c.data.dim, j) = (double)c.confidence(i, j);
            }
    }

    template <typename T>
    void extract_explanation(Component<T> const& c)
    {
        A.resize(1);
        A[0].clear();

        if (c.summary.size() > c.data.dim)
        {
            set.reserve(c.summary.size() - c.data.dim);

            for (size_t i = c.data.dim; i < c.summary.size(); ++i)
            {
                if (count(c.summary.point(i)) > 0)
                {
                    auto& next = set.emplace_back();
                    next.reserve(count(c.summary.point(i)));
                    foreach (c.summary.point(i), [&](size_t index) { next.push_back(index); })
                        ;
                }
            }
            // this->frequency.resize({set.size()});
            // auto target = static_cast<double*>(this->frequency.request().ptr);
            // std::copy(c.summary.template col<0>().begin() + c.data.dim,
            //           c.summary.template col<0>().end(),
            //           target);
            A.back().container.resize(c.summary.size() - c.data.dim, 1);
        }

        this->confidence_matrix.resize(c.confidence.size() - c.data.dim, 1);
        for (size_t i = c.data.dim; i < c.confidence.size(); ++i)
        {
            confidence_matrix(i - c.data.dim) = (double)c.confidence[i];
        }
    }
};

class Interface : public Explaination
{
public:
    virtual void fit(Dataset<tag_sparse> const& x, std::vector<size_t> const& y) = 0;
    virtual void fit(Dataset<tag_dense> const& x, std::vector<size_t> const& y)  = 0;
    virtual label_vector predict(Dataset<tag_sparse> const& x) const = 0;
    virtual label_vector predict(Dataset<tag_dense> const& x) const  = 0;
    virtual real_matrix predict_probabilities(Dataset<tag_sparse> const& x) const = 0;
    virtual real_matrix predict_probabilities(Dataset<tag_dense> const& x) const  = 0;
    virtual real_matrix predict_log_probabilities(Dataset<tag_sparse> const& x) const = 0;
    virtual real_matrix predict_log_probabilities(Dataset<tag_dense> const& x) const  = 0;
    virtual double log_likelihood(Dataset<tag_sparse> const& x,
                                  std::vector<size_t> const& y) const = 0;
    virtual double log_likelihood(Dataset<tag_dense> const&  x,
                                  std::vector<size_t> const& y) const = 0;
    virtual ~Interface() = default;
};

template <typename Trait, typename BaseClass>
class Single_class : public BaseClass
{
public:
    using pattern_type = typename Trait::pattern_type;

    virtual void do_fit(Dataset<pattern_type> x) = 0;

    label_vector do_predict(Dataset<pattern_type> const& x) const
    {
        return label_vector(x.size());
    }
    real_matrix do_predict_probabilities(Dataset<pattern_type> const& x) const
    {
        const auto& pr = component.model;
        real_matrix y(x.size(), 1);
#pragma omp parallel for schedule(dynamic, 1)
        for (size_t i = 0; i < x.size(); ++i) { y(i, 0) = (double)pr.expectation(x.point(i)); }
        return y;
    }
    real_matrix do_predict_log_probabilities(Dataset<pattern_type> const& x) const
    {
        const auto& pr = component.model;
        real_matrix y(x.size(), 1);
#pragma omp parallel for schedule(dynamic, 1)
        for (size_t i = 0; i < x.size(); ++i)
        {
            y(i, 0) = (double)pr.log_expectation(x.point(i));
        }
        return y;
    }
    auto do_log_likelihood(Dataset<pattern_type> const& x) const
    {
        const auto&                pr = component.model;
        typename Trait::float_type l  = 0;
#pragma omp parallel for schedule(dynamic, 1)
        for (size_t i = 0; i < x.size(); ++i) { l += pr.log_expectation(x.point(i)); }
        return static_cast<double>(l);
    }
    //
    // Interface
    //

    void fit(Dataset<tag_sparse> const& x, std::vector<size_t> const& y) override
    {
        do_fit(to_dataset<pattern_type>(x));
    }
    void fit(Dataset<tag_dense> const& x, std::vector<size_t> const& y) override
    {
        do_fit(to_dataset<pattern_type>(x));
    }
    label_vector predict(Dataset<tag_sparse> const& x) const override
    {
        return do_predict(to_dataset<pattern_type>(x));
    }
    label_vector predict(Dataset<tag_dense> const& x) const override
    {
        return do_predict(to_dataset<pattern_type>(x));
    }
    real_matrix predict_probabilities(Dataset<tag_sparse> const& x) const override
    {
        return do_predict_probabilities(to_dataset<pattern_type>(x));
    }
    real_matrix predict_probabilities(Dataset<tag_dense> const& x) const override
    {
        return do_predict_probabilities(to_dataset<pattern_type>(x));
    }
    real_matrix predict_log_probabilities(Dataset<tag_sparse> const& x) const override
    {
        return do_predict_log_probabilities(to_dataset<pattern_type>(x));
    }
    real_matrix predict_log_probabilities(Dataset<tag_dense> const& x) const override
    {
        return do_predict_log_probabilities(to_dataset<pattern_type>(x));
    }
    double log_likelihood(Dataset<tag_sparse> const& x,
                          std::vector<size_t> const& y) const override
    {
        return do_log_likelihood(to_dataset<pattern_type>(x));
    }
    double log_likelihood(Dataset<tag_dense> const&  x,
                          std::vector<size_t> const& y) const override
    {
        return do_log_likelihood(to_dataset<pattern_type>(x));
    }

    //
    // Disable Explanation
    //
    std::vector<size_t> contrasting(std::vector<size_t> const&,
                                    std::vector<size_t> const&) const override
    {
        throw std::runtime_error("A single class cannot contrast itself");
    };
    std::vector<size_t> emerging(std::vector<size_t> const&, std::vector<size_t> const&) const
    {
        throw std::runtime_error("A single class cannot contrast itself");
    }
    std::vector<size_t> common(std::vector<size_t> const&) const
    {
        throw std::runtime_error("A single class cannot contrast itself");
    }
    std::vector<size_t> characteristic(int) const
    {
        throw std::runtime_error("A single class cannot contrast itself");
    }
    std::vector<size_t> unique(std::vector<size_t> const&) const
    {
        throw std::runtime_error("A single class cannot contrast itself");
    }

protected:
    Component<Trait> component;
};

template <typename Trait, typename BaseClass>
class Multi_class : public BaseClass
{
public:
    using pattern_type = typename Trait::pattern_type;
    using float_type   = typename Trait::float_type;

    virtual void do_fit(Dataset<pattern_type> x, std::vector<size_t> const& y) = 0;

    label_vector do_predict(Dataset<pattern_type> const& x) const
    {
        label_vector y(x.size());

        using float_type =
            std::decay_t<decltype(composition.models[0].expectation(x.point(0)))>;

        const auto k = composition.models.size();

        std::vector<size_t> labels(k, 0);
        for (size_t i = 0; i < k; ++i) { labels[i] = label(composition.data.subset(i)[0]); }

        for (size_t i = 0; i < x.size(); ++i)
        {
            std::pair<size_t, float_type> y_i = {};
            for (size_t j = 0; j < k; ++j)
            {
                auto p = composition.models[j].expectation(x.point(i));
                if (y_i.second < p) { y_i = {j, p}; }
                y[i] = labels[y_i.first];
            }
        }

        return y;
    }
    real_matrix do_predict_probabilities(Dataset<pattern_type> const& x) const
    {
        const auto& pr = composition.models;
        real_matrix y(x.size(), pr.size());
#pragma omp parallel for schedule(dynamic, 1) collapse(2)
        for (size_t i = 0; i < x.size(); ++i)
        {
            for (size_t j = 0; j < pr.size(); ++j)
            {
                y(i, j) = (double)pr[j].expectation(x.point(i));
            }
        }

        return y;
    }
    real_matrix do_predict_log_probabilities(Dataset<pattern_type> const& x) const
    {
        const auto& pr = composition.models;
        real_matrix y(x.size(), pr.size());
#pragma omp parallel for schedule(dynamic, 1) collapse(2)
        for (size_t i = 0; i < x.size(); ++i)
        {
            for (size_t j = 0; j < pr.size(); ++j)
            {
                y(i, j) = (double)pr[j].log_expectation(x.point(i));
            }
        }

        return y;
    }
    auto do_log_likelihood(Dataset<pattern_type> const& x, std::vector<size_t> const& y) const
    {
        const auto&                pr = composition.models;
        typename Trait::float_type l  = 0;
#pragma omp parallel for schedule(dynamic, 1)
        for (size_t i = 0; i < x.size(); ++i) { l += pr[y[i]].log_expectation(x.point(i)); }
        return static_cast<double>(l);
    }
    //
    // Interface
    //

    void fit(Dataset<tag_sparse> const& x, std::vector<size_t> const& y) override
    {
        do_fit(to_dataset<pattern_type>(x), y);
    }
    void fit(Dataset<tag_dense> const& x, std::vector<size_t> const& y) override
    {
        do_fit(to_dataset<pattern_type>(x), y);
    }
    label_vector predict(Dataset<tag_sparse> const& x) const override
    {
        return do_predict(to_dataset<pattern_type>(x));
    }
    label_vector predict(Dataset<tag_dense> const& x) const override
    {
        return do_predict(to_dataset<pattern_type>(x));
    }
    real_matrix predict_probabilities(Dataset<tag_sparse> const& x) const override
    {
        return do_predict_probabilities(to_dataset<pattern_type>(x));
    }
    real_matrix predict_probabilities(Dataset<tag_dense> const& x) const override
    {
        return do_predict_probabilities(to_dataset<pattern_type>(x));
    }
    real_matrix predict_log_probabilities(Dataset<tag_sparse> const& x) const override
    {
        return do_predict_log_probabilities(to_dataset<pattern_type>(x));
    }
    real_matrix predict_log_probabilities(Dataset<tag_dense> const& x) const override
    {
        return do_predict_log_probabilities(to_dataset<pattern_type>(x));
    }
    double log_likelihood(Dataset<tag_sparse> const& x,
                          std::vector<size_t> const& y) const override
    {
        return do_log_likelihood(to_dataset<pattern_type>(x), y);
    }
    double log_likelihood(Dataset<tag_dense> const&  x,
                          std::vector<size_t> const& y) const override
    {
        return do_log_likelihood(to_dataset<pattern_type>(x), y);
    }

protected:
    Composition<Trait> composition;
};

template <typename Trait>
class Desc_single_class_impl : public Single_class<Trait, Interface>
{
public:
    using pattern_type = typename Trait::pattern_type;
    using base_type    = Single_class<Trait, Interface>;

    void do_fit(Dataset<pattern_type> x) override
    {
        this->component      = Component<Trait>();
        this->component.data = std::move(x);
        initialize_model(this->component, this->cfg);
        this->elapsed =
            timeit([&] { discover_patterns_generic(this->component, this->cfg, IDesc{}); });
        this->extract_explanation(this->component);
    }
};

template <typename Trait>
class Desc_impl : public Multi_class<Trait, Interface>
{
public:
    using pattern_type = typename Trait::pattern_type;
    using base_type    = Single_class<Trait, Interface>;

    void do_fit(Dataset<pattern_type> x, std::vector<size_t> const& y) override
    {
        this->composition      = Composition<Trait>();
        this->composition.data = PartitionedData(std::move(x), y);
        initialize_model(this->composition, this->cfg);
        this->elapsed =
            timeit([&] { discover_patterns_generic(this->composition, this->cfg, IDesc{}); });
        this->extract_explanation(this->composition);
    }
};

// template <typename Trait>
// class DescMDL_single_class_impl : public Single_class<Trait, Interface>
// {
// public:
//     using pattern_type = typename Trait::pattern_type;
//     using base_type    = Single_class<Trait, Interface>;

//     void do_fit(Dataset<pattern_type> x) override
//     {
//         this->component      = Component<Trait>();
//         this->component.data = std::move(x);
//         initialize_model(this->component, this->cfg);
//         this->elapsed =
//             timeit([&] { discover_patterns_generic(this->component, this->cfg, IDescMDL{});
//             });
//         this->extract_explanation(this->component);
//     }
// };

// template <typename Trait>
// class DescMDL_impl : public Multi_class<Trait, Interface>
// {
// public:
//     using pattern_type = typename Trait::pattern_type;
//     using base_type    = Single_class<Trait, Interface>;

//     void do_fit(Dataset<pattern_type> x, std::vector<size_t> const& y) override
//     {
//         this->composition      = Composition<Trait>();
//         this->composition.data = PartitionedData(std::move(x), y);
//         initialize_model(this->composition, this->cfg);
//         this->elapsed = timeit(
//             [&] { discover_patterns_generic(this->composition, this->cfg, IDescMDL{}); });
//         this->extract_explanation(this->composition);
//     }
// };

template <typename S, typename T, typename Fn>
void select_dist_type(Fn&& fn)
{
    using trait = Trait<S, T, MaxEntDistribution<S, T>>;
    std::forward<Fn>(fn)(trait{});
}

// using precise_float_t = long double;

template <typename S, typename Fn>
void select_real_type(bool is_precise, Fn&& fn)
{
    if (is_precise) { select_dist_type<S, precise_float_t>(std::forward<Fn>(fn)); }
    else
    {
        select_dist_type<S, double>(std::forward<Fn>(fn));
    }
}

template <typename Fn>
void build_trait(bool is_sparse, bool is_precise, Fn&& fn)
{
    if (is_sparse) { select_real_type<tag_sparse>(is_precise, std::forward<Fn>(fn)); }
    else
    {
        select_real_type<tag_dense>(is_precise, std::forward<Fn>(fn));
    }
}

PyObject* to_numpy(real_matrix const& x)
{
    // double* data = new double[x.rows() * x.cols()];
    double* data = new double[x.size()];
    std::copy(x.data(), x.data() + x.size(), data);
    _import_array();
    // npy_intp  dims[2] = {x.rows(), x.cols()};
    npy_intp dims[2] = {static_cast<npy_intp>(x.extent(0)), static_cast<npy_intp>(x.extent(1))};
    PyObject* o      = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, static_cast<void*>(data));
    PyArray_ENABLEFLAGS((PyArrayObject*)o, NPY_ARRAY_OWNDATA);
    return o;
}

class CppDesc
{
public:
    bool   sparse      = false;
    bool   mdl         = true;
    bool   beam_search = true;
    size_t min_support = 2;

#if HAS_HIGH_PRECISION_FLOAT_TYPE
    bool high_precision = false;
#else
    constexpr static bool high_precision = false;
#endif

    template <class T>
    void fit(Dataset<T> const& x, std::vector<size_t> const& y = {})
    {
        setup(x, y);
    }
    template <class T>
    std::vector<size_t> predict(Dataset<T> const& x) const
    {
        auto y = get().predict(x);
        return std::vector<size_t>(y.begin(), y.end());
    }
    template <class T>
    auto predict_probabilities(Dataset<T> const& x) const
    {
        return to_numpy(get().predict_probabilities(x));
    }
    template <class T>
    auto predict_log_probabilities(Dataset<T> const& x) const
    {
        return to_numpy(get().predict_log_probabilities(x));
    }
    template <class T>
    double log_likelihood(Dataset<T> const& x, std::vector<size_t> const& y) const
    {
        return get().log_likelihood(x, y);
    }

    std::vector<size_t> contrasting(std::vector<size_t> const& s, std::vector<size_t> const& t)
    {
        return get().contrasting(s, t);
    }
    std::vector<size_t> emerging(std::vector<size_t> const& s, std::vector<size_t> const& t)
    {
        return get().emerging(s, t);
    }
    std::vector<size_t> common(std::vector<size_t> const& s) { return get().common(s); }
    std::vector<size_t> characteristic(int i) { return get().characteristic(i); }
    std::vector<size_t> unique(std::vector<size_t> const& s) { return get().unique(s); }
    const std::vector<std::vector<size_t>>& patterns() const { return get().patterns(); }
    std::chrono::milliseconds               elapsed() const { return get().elapsed; }
    auto   confidence() const { return to_numpy(get().confidence_matrix); }
    size_t elapsed_milliseconds() const { return get().elapsed.count(); }

    // auto independent_log_pvalues() {}

private:
    std::unique_ptr<Interface> impl;

    Interface& get()
    {
        if (impl) return *impl;
        throw std::runtime_error("Model is not yet fitted!");
    }
    const Interface& get() const
    {
        if (impl) return *impl;
        throw std::runtime_error("Model is not yet fitted!");
    }

    template <typename pattern_type, typename float_type>
    void do_setup(bool use_single_class //, bool relaxed
    )
    {
        // if (relaxed)
        // {
        //     using dist_t     = RelEntDistribution<pattern_type, float_type>;
        //     using trait_type = Trait<pattern_type, float_type, dist_t>;

        //     if (use_single_class)
        //         impl = std::make_unique<Desc_single_class_impl<trait_type>>();
        //     else
        //         impl = std::make_unique<Desc_impl<trait_type>>();
        // }
        // else
        {
            using dist_t     = MaxEntDistribution<pattern_type, float_type>;
            using trait_type = Trait<pattern_type, float_type, dist_t>;

            if (use_single_class)
                impl = std::make_unique<Desc_single_class_impl<trait_type>>();
            else
                impl = std::make_unique<Desc_impl<trait_type>>();
        }
    }

    template <typename X>
    void setup(X const& x, std::vector<size_t> const& y)
    {
        bool use_single_class = y.empty();

        build_trait(sparse, high_precision, [&](auto t) {
            using Trait = std::decay_t<decltype(t)>;
            // if (mdl)
            // {
            //     if (use_single_class)
            //         impl = std::make_unique<DescMDL_single_class_impl<Trait>>();
            //     else
            //         impl = std::make_unique<DescMDL_impl<Trait>>();
            // }
            // else
            {
                if (use_single_class)
                    impl = std::make_unique<Desc_single_class_impl<Trait>>();
                else
                    impl = std::make_unique<Desc_impl<Trait>>();
            }
        });

        get().cfg.min_support = min_support;
        if (!beam_search) get().cfg.search_depth = 1;

        get().fit(x, y);
    }
};

} // namespace sd::disc
