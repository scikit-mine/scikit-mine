#pragma once

#include <cstddef>

namespace sd::viva
{

size_t bit_permute_step(size_t x, size_t m, int shift)
{
    size_t t;
    t = ((x >> shift) ^ x) & m;
    x = (x ^ t) ^ (t << shift);
    return x;
}

size_t next_permutation(size_t v)
{
    size_t t = (v | (v - 1)) + 1;
    size_t w = t | ((((t & -t) / (v & -v)) >> 1) - 1);
    return w;
}

template <typename Fn>
void permute_level(size_t n, size_t k, Fn&& fn)
{
    size_t i = 0;
    for (size_t z = 0; z < k; ++z)
        i |= (1 << z);

    size_t max = i << (n - k);
    do
    {
        fn(i);
        i = next_permutation(i);
    } while (i != max);
    fn(i); // for max
}

template <typename Fn>
void permute_all(size_t n, Fn&& fn)
{
    size_t i = 0;
    for (size_t z = 0; z < n; ++z)
        i |= (1 << z);
    fn(i); // for highest

    for (size_t s = n; s-- > 1;)
        permute_level(n, s, fn);
    fn(size_t(0)); // emptyset
}

template <typename Fn>
void permute_level_limit(size_t n, size_t k, Fn&& fn, size_t& limit)
{

    size_t i = 0;
    for (size_t z = 0; z < k; ++z)
        i |= (1 << z);

    size_t max = i << (n - k);
    do
    {
        fn(i);
        i = next_permutation(i);

        if (limit-- == 0)
            return;

    } while (i != max);
    fn(i); // for max
}

template <typename Fn>
void permute_first_n(size_t n, size_t first_n, Fn&& fn)
{
    if (first_n == 0)
        return;

    size_t i = 0;
    for (size_t z = 0; z < n; ++z)
        i |= (1 << z);

    fn(i); // for highest

    --first_n;

    for (size_t s = n; s-- > 1 && first_n > 0;)
        permute_level_limit(n, s, fn, first_n);
    fn(size_t(0)); // emptyset
}

template <typename Fn>
void foreach(size_t i, Fn&& fn)
{
    size_t one = 1;
    for (size_t j = 0, l = i; l != 0; ++j, l >>= one)
    { /*&& j < bound*/
        if (l & one)
        {
            fn(j);
        }
    }
}

} // namespace sd::viva
