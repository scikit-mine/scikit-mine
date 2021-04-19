from libcpp.vector cimport vector
from libcpp cimport bool
import numpy as np

cdef extern from "<bindings/cython/Desc.hxx>" namespace "sd::disc":
    cdef cppclass tag_sparse
    cdef cppclass tag_dense
    cdef cppclass itemset[T]:
        void insert(size_t x)
        void clear()
    cdef cppclass Dataset[T]:
        void insert(itemset[T] x)
    cdef cppclass CppDesc:
        CppDesc()
        bool sparse
        bool high_precision_float #if enabled
        bool beam_search
        size_t min_support
        void fit(Dataset[tag_sparse] x, vector[size_t] y)
        void fit(Dataset[tag_dense] x, vector[size_t] y)
        vector[size_t] predict(Dataset[tag_sparse] x)
        vector[size_t] predict(Dataset[tag_dense] x)
        object predict_probabilities(Dataset[tag_sparse] x)
        object predict_probabilities(Dataset[tag_dense] x)
        object predict_log_probabilities(Dataset[tag_sparse] x)
        object predict_log_probabilities(Dataset[tag_dense] x)
        double log_likelihood(Dataset[tag_sparse] x, vector[size_t] y)
        double log_likelihood(Dataset[tag_dense] x, vector[size_t] y)
        object confidence()
        size_t elapsed_milliseconds()
        vector[size_t] common(vector[size_t])
        vector[size_t] characteristic(size_t)
        vector[size_t] contrasting(vector[size_t], vector[size_t])
        vector[size_t] emerging(vector[size_t], vector[size_t])
        vector[size_t] unique(vector[size_t], vector[size_t])
        vector[vector[size_t]] patterns()
        