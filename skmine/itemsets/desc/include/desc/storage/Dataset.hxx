#pragma once

#include <desc/storage/Itemset.hxx>

#include <datatable/data_table.hxx>

namespace sd::disc
{

template <typename T>
decltype(auto) point(std::tuple<T>&& t)
{
    return std::get<0>(t);
}
template <typename T>
decltype(auto) point(std::tuple<T>& t)
{
    return std::get<0>(t);
}
template <typename T>
decltype(auto) point(const std::tuple<T>& t)
{
    return std::get<0>(t);
}
template <typename... Ts>
decltype(auto) point(std::tuple<Ts...>&& t)
{
    return std::get<1>(t);
}
template <typename... Ts>
decltype(auto) point(std::tuple<Ts...>& t)
{
    return std::get<1>(t);
}
template <typename... Ts>
decltype(auto) point(const std::tuple<Ts...>& t)
{
    return std::get<1>(t);
}
template <typename... Ts>
decltype(auto) label(std::tuple<Ts...>&& t)
{
    return std::get<0>(t);
}
template <typename... Ts>
decltype(auto) label(std::tuple<Ts...>& t)
{
    return std::get<0>(t);
}
template <typename... Ts>
decltype(auto) label(const std::tuple<Ts...>& t)
{
    return std::get<0>(t);
}
} // namespace sd::disc

namespace sd::df
{

template <>
struct column_type<disc::tag_dense>
{
    using value = std::vector<disc::storage_container<disc::tag_dense>>;
};
template <>
struct column_type<disc::tag_sparse>
{
    using value = std::vector<disc::storage_container<disc::tag_sparse>>;
};

} // namespace sd::df

namespace sd::disc
{

// template <typename S>
// struct PartitionedData;

template <typename S>
struct Dataset : public sd::df::col_store<S>
{
    using pattern_type = S;

    template <typename T>
    void insert(T&& t)
    {
        this->push_back(storage_container<S>(std::forward<T>(t)));
        if constexpr (std::is_same_v<S, tag_sparse>)
        {
            point(this->size() - 1).container.shrink_to_fit();
        }
        dim = std::max(dim, get_dim(point(this->size() - 1)));
    }

    void insert(itemset<pattern_type> const & t)
    {
        this->insert(storage_container<S>(t));
    }

    void insert(size_t t)
    {
        storage_container<S> buf;
        buf.insert(t);
        this->insert(std::move(buf));
    }

    template <typename T, typename = std::enable_if_t<!std::is_same_v<T, S>>>
    explicit Dataset(T&& other)
    {
        this->reserve(other.size());
        storage_container<S> row;
        for (auto&& o : other)
        {
            using sd::disc::point;
            row.clear();            
            sd::foreach(point(o), [&](size_t i) { row.insert(i); });
            this->insert(row);
        }
    }

    Dataset()               = default;
    Dataset(Dataset&&)      = default;
    Dataset(const Dataset&) = default;
    Dataset& operator=(const Dataset&) = default;
    Dataset& operator=(Dataset&&) = default;

    decltype(auto) point(size_t index) const { return this->template col<0>()[index]; }
    decltype(auto) point(size_t index) { return this->template col<0>()[index]; }

    size_t capacity() const { return this->template col<0>().capacity(); }

    size_t dim = 0;
};

template <typename L, typename S>
struct LabeledDataset : public sd::df::col_store<L, S>
{
    using label_type   = L;
    using pattern_type = S;

    template <typename T>
    void insert(const L& label, T&& t)
    {
        storage_container<S> buf;
        buf.insert(t);
        this->push_back(label, std::move(buf));
        if constexpr (std::is_same_v<S, tag_sparse>)
        {
            point(this->size() - 1).container.shrink_to_fit();
        }
        dim = std::max(dim, get_dim(point(this->size() - 1)));
    }
    template <typename T>
    void insert(T&& t)
    {
        insert(L{}, std::forward<T>(t));
    }
    const auto&    labels() const { return this->template col<0>(); }
    decltype(auto) point(size_t index) const { return this->template col<1>()[index]; }
    decltype(auto) label(size_t index) const { return this->template col<0>()[index]; }
    decltype(auto) point(size_t index) { return this->template col<1>()[index]; }
    decltype(auto) label(size_t index) { return this->template col<0>()[index]; }

    size_t dim = 0;
};

template <typename S>
struct PartitionedData_ : public sd::df::col_store<size_t, S, size_t>
{
    using pattern_type = S;

    PartitionedData_() = default;

    PartitionedData_(Dataset<S>&& rhs)
    {
        this->reserve(rhs.size());
        dim = rhs.dim;
        for (size_t i = 0; i < rhs.size(); ++i)
        {
            this->push_back(size_t(0), std::move(rhs.point(i)), i);
        }
        group_by_label();
    }

    PartitionedData_(Dataset<S>&& rhs, const std::vector<size_t>& labels)
    {
        this->reserve(rhs.size());
        dim = rhs.dim;
        for (size_t i = 0; i < rhs.size(); ++i)
        {
            this->push_back(labels[i], std::move(rhs.point(i)), i);
        }
        group_by_label();
    }

    auto subset(size_t index)
    {
        return this->map_cols([&](auto& s) {
            return sd::make_cpslice(s).cut(positions[index], positions[index + 1]);
        });
    }

    auto subset(size_t index) const
    {
        return this->map_cols([&](auto& s) {
            return sd::make_cpslice(s).cut(positions[index], positions[index + 1]);
        });
    }

    size_t num_components() const { return positions.empty() ? 0 : positions.size() - 1; }

    const auto& elements() const { return this->template col<1>(); }

    void revert_order()
    {
        std::sort(this->begin(), this->end(), [](const auto& a, const auto& b) {
            return get<2>(a) < get<2>(b);
        });
        positions.clear();
    }

    void group_by_label()
    {
        auto lt = [](const auto& a, const auto& b) { return get<0>(a) < get<0>(b); };

        positions.clear();

        std::sort(this->begin(), this->end(), lt);

        size_t last_label = std::numeric_limits<size_t>::max();
        auto&  labels     = this->template col<0>();
        for (size_t i = 0; i < labels.size(); ++i)
        {
            const auto& l = labels[i];
            if (last_label != l)
            {
                last_label = l;
                positions.push_back(i);
            }
        }
        positions.push_back(this->size());
        num_components_backup = num_components();
    }

    void reserve(size_t n)
    {
        this->foreach_col([n](auto& c) { c.reserve(n); });
    }

    decltype(auto) point(size_t index) { return this->template col<1>()[index]; }
    decltype(auto) point(size_t index) const { return this->template col<1>()[index]; }
    decltype(auto) label(size_t index) const { return this->template col<0>()[index]; }
    decltype(auto) label(size_t index) { return this->template col<0>()[index]; }
    decltype(auto) original_position(size_t index) const
    {
        return this->template col<2>()[index];
    }

    std::vector<size_t> positions;
    size_t              dim                   = 0;
    size_t              num_components_backup = 0;
};

template <typename T>
using itemset_view = std::conditional_t<is_sparse(T{}),
                                        sd::sparse_bit_view<sd::slice<const sparse_index_type>>,
                                        sd::bit_view<sd::slice<const uint64_t>>>;

itemset_view<tag_sparse> make_view(const storage_container<tag_sparse>& x)
{
    return {x.container};
}
itemset_view<tag_dense> make_view(const storage_container<tag_dense>& x)
{
    return {{x.container}, x.length_};
}
template <typename T>
void swap(itemset_view<T>& a, itemset_view<T>& b)
{
    std::swap(a, b);
}

template <typename S>
struct PartitionedData : public sd::df::col_store<size_t, itemset_view<S>, size_t>
{
    using pattern_type = S;

    PartitionedData() = default;
    PartitionedData(const PartitionedData&) = default;
    PartitionedData(PartitionedData&&)      = default;
    PartitionedData& operator=(const PartitionedData&) = default;
    PartitionedData& operator=(PartitionedData&&) = default;

    PartitionedData(Dataset<S>&& rhs)
    {
        data = std::make_shared<Dataset<S>>(std::forward<Dataset<S>>(rhs));
        dim  = data->dim;
        this->resize(data->size());

        auto& ol = this->template col<2>();
        std::iota(ol.begin(), ol.end(), 0);

        reindex();
        group_by_label();
    }

    PartitionedData(Dataset<S>&& rhs, const std::vector<size_t>& labels)
    {
        data = std::make_shared<Dataset<S>>(std::forward<Dataset<S>>(rhs));

        assert(data->size() == labels.size());

        dim  = data->dim;
        this->resize(data->size());

        auto& ol = this->template col<2>();
        std::iota(ol.begin(), ol.end(), 0);

        std::copy_n(labels.begin(), labels.size(), this->template col<0>().begin());

        reindex();
        group_by_label();
    }

    auto subset(size_t index)
    {
        return this->map_cols([&](auto& s) {
            return sd::make_cpslice(s).cut(positions[index], positions[index + 1]);
        });
    }

    auto subset(size_t index) const
    {
        return this->map_cols([&](auto& s) {
            return sd::make_cpslice(s).cut(positions[index], positions[index + 1]);
        });
    }

    size_t num_components() const { return positions.empty() ? 0 : positions.size() - 1; }

    const auto& elements() const { return this->template col<1>(); }

    void reindex()
    {
        assert(data->size() == this->size());
        for (size_t i = 0; i < this->size(); ++i)
        {
            point(i) = make_view(data->point(i));
            assert(sd::equal(point(i), data->point(i)));
        }
    }

    void revert_order()
    {
        std::sort(this->begin(), this->end(), [](const auto& a, const auto& b) {
            return get<2>(a) < get<2>(b);
        });
        positions.clear();
    }

    void group_by_label()
    {
        auto lt = [](const auto& a, const auto& b) { return get<0>(a) < get<0>(b); };

        positions.clear();

        std::sort(this->begin(), this->end(), lt);

        size_t last_label = std::numeric_limits<size_t>::max();
        auto&  labels     = this->template col<0>();
        for (size_t i = 0; i < labels.size(); ++i)
        {
            const auto& l = labels[i];
            if (last_label != l)
            {
                last_label = l;
                positions.push_back(i);
            }
        }
        positions.push_back(this->size());
        num_components_backup = num_components();
    }

    void reserve(size_t n)
    {
        this->foreach_col([n](auto& c) { c.reserve(n); });
        this->data->reserve(n);
    }

    decltype(auto) point(size_t index) { return this->template col<1>()[index]; }
    decltype(auto) point(size_t index) const { return this->template col<1>()[index]; }
    decltype(auto) label(size_t index) const { return this->template col<0>()[index]; }
    decltype(auto) label(size_t index) { return this->template col<0>()[index]; }
    decltype(auto) original_position(size_t index) const
    {
        return this->template col<2>()[index];
    }

    const auto& underlying_data() const { return *data; }

// private:
    std::shared_ptr<Dataset<S>> data;
    std::vector<size_t> positions;

public:
    size_t dim                   = 0;
    size_t num_components_backup = 0;
};

template <typename S>
void simplify_labels(PartitionedData<S>& data)
{
    for (size_t subset = 0, n = data.num_components(); subset < n; ++subset)
    {
        for (auto [y, _1, _2] : data.subset(subset))
        {
            y = subset;
        }
    }
}

} // namespace sd::disc