

cdef Dataset[tag_dense] to_dense_dataset(xs):
    cdef Dataset[tag_dense] data
    cdef itemset[tag_dense] t
    for x in xs: 
        t.clear()
        for i in x: t.insert(i)
        data.insert(t)
    return data

cdef Dataset[tag_sparse] to_sparse_dataset(xs):
    cdef Dataset[tag_sparse] data
    cdef itemset[tag_sparse] t
    for x in xs: 
        t.clear()
        for i in x: t.insert(i)
        data.insert(t) 
    return data

cdef vector[size_t] to_size_t_vector(x):
    cdef vector[size_t] y
    y.reserve(len(x))
    for i in x: y.push_back(i)
    return y

cdef class Desc: 
    is_sparse = False
    use_high_precision_floats = False
    min_support = 2
    cdef CppDesc desc

    def fit(self, D, y=None):
        # self.has_labels = y is not None and len(y) > 0
        if self.is_sparse:
            self.desc.fit(to_sparse_dataset(D), to_size_t_vector(y))
        else:
            self.desc.fit(to_dense_dataset(D), to_size_t_vector(y))
    def predict(self, D): 
        return self.desc.predict(to_sparse_dataset(D)) if self.is_sparse else self.desc.predict(to_dense_dataset(D))
    def predict_log_probabilities(self, D): 
        return self.desc.predict_log_probabilities(to_sparse_dataset(D)) if self.is_sparse else self.desc.predict_log_probabilities(to_dense_dataset(D))
    def predict_probabilities(self, D): 
        return self.desc.predict_probabilities(to_sparse_dataset(D)) if self.is_sparse else self.desc.predict_probabilities(to_dense_dataset(D))
    def log_likelihood(self, D, y): 
        return self.desc.log_likelihood(to_sparse_dataset(D), to_size_t_vector(y)) if self.is_sparse else self.desc.log_likelihood(to_dense_dataset(D), to_size_t_vector(y))
    def common(self, i): return self.desc.common(i)
    def characteristic(self, i): return self.desc.characteristic(i)
    def emerging(self, i, j): return self.desc.emerging(i, j)
    def contrasting(self, i, j): return self.desc.contrasting(i, j)
    def confidence(self): return self.desc.confidence()
