#pragma once
#include <desc/storage/Dataset.hxx>

namespace sd
{

template <typename T>
void increment(sd::bit_view<T>& in)
{
    for (size_t i = 0; i < in.length(); ++i)
    {
        if (!in.test(i))
        {
            in.set(i, true);
            break;
        }
        in.set(i, false);
    }
}

template <typename T>
void decrement(sd::bit_view<T>& in)
{
    for (size_t i = in.length(); i-- > 0;)
    {
        if (in.test(i))
        {
            in.set(i, false);
            break;
        }
        in.set(i, true);
        // if ((in[i] ^= true) == false) {
        //     break;
        // }
    }
}

} // namespace sd