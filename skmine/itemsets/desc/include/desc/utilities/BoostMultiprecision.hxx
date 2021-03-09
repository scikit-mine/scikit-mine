#pragma once
#ifndef INTERFACE_BOOST_MULTIPRECISION
#define INTERFACE_BOOST_MULTIPRECISION

#if __has_include(<quadmath.h>) && __has_include(<boost/multiprecision/float128.hpp>) && WITH_QUADMATH
// #pragma message "float128"

#define HAS_HIGH_PRECISION_FLOAT_TYPE 1

#include <boost/multiprecision/float128.hpp>
#include <limits>

using precise_float_t = boost::multiprecision::float128;
// using precise_float_t = __float128;

namespace sd::disc
{
constexpr const char* float_storage_type_to_str(boost::multiprecision::float128)
{
    return "float128";
}
} // namespace sd::disc

#pragma omp declare reduction \
  (*:precise_float_t:omp_out=omp_out*omp_in) \
  initializer(omp_priv=1)

#pragma omp declare reduction \
  (+:precise_float_t:omp_out=omp_out+omp_in) \
  initializer(omp_priv=0)

namespace std
{
auto log2(const precise_float_t& x) { return boost::multiprecision::log2(x); }
auto log(const precise_float_t& x) { return boost::multiprecision::log(x); }
auto abs(const precise_float_t& x) { return boost::multiprecision::abs(x); }
auto exp2(const precise_float_t& x) { return boost::multiprecision::exp2(x); }
auto exp(const precise_float_t& x) { return boost::multiprecision::exp(x); }
auto isnan(const precise_float_t& x) { return boost::multiprecision::isnan(x); }
auto isinf(const precise_float_t& x) { return boost::multiprecision::isinf(x); }
auto sqrt(const precise_float_t& x) { return boost::multiprecision::sqrt(x); }
} // namespace std

#elif __has_include(<mpfr.h>) && __has_include(<boost/multiprecision/mpfr.hpp>) && WITH_MPFR
// #pragma message "mpfr"

#define HAS_HIGH_PRECISION_FLOAT_TYPE 1

#include <boost/multiprecision/mpfr.hpp>

using precise_float_t = boost::multiprecision::static_mpfr_float_100;
namespace sd::disc
{
constexpr const char* float_storage_type_to_str(boost::multiprecision::static_mpfr_float_100)
{
    return "static_mpfr_float_100";
}
} // namespace sd::disc

#pragma omp declare reduction \
  (*:precise_float_t:omp_out=omp_out*omp_in) \
  initializer(omp_priv=1)

#pragma omp declare reduction \
  (+:precise_float_t:omp_out=omp_out+omp_in) \
  initializer(omp_priv=0)

namespace std
{
auto log2(const precise_float_t& x) { return boost::multiprecision::log2(x); }
auto log(const precise_float_t& x) { return boost::multiprecision::log(x); }
auto abs(const precise_float_t& x) { return boost::multiprecision::abs(x); }
auto exp2(const precise_float_t& x) { return boost::multiprecision::exp2(x); }
auto exp(const precise_float_t& x) { return boost::multiprecision::exp(x); }
auto isnan(const precise_float_t& x) { return boost::multiprecision::isnan(x); }
auto isinf(const precise_float_t& x) { return boost::multiprecision::isinf(x); }
auto sqrt(const precise_float_t& x) { return boost::multiprecision::sqrt(x); }
} // namespace std

#else

// #pragma message "long double"

using precise_float_t = long double;

#endif

// #if HAS_HIGH_PRECISION_FLOAT_TYPE

// #pragma omp declare reduction (*:precise_float_t:omp_out=omp_out*omp_in)
// initializer(omp_priv=1) #pragma omp declare reduction
// (+:precise_float_t:omp_out=omp_out+omp_in) initializer(omp_priv=0)

// namespace std
// {
// auto log2(const precise_float_t& x) { return boost::multiprecision::log2(x); }
// auto abs(const precise_float_t& x) { return boost::multiprecision::abs(x); }
// auto exp2(const precise_float_t& x) { return boost::multiprecision::exp2(x); }
// auto isnan(const precise_float_t& x) { return boost::multiprecision::isnan(x); }
// auto isinf(const precise_float_t& x) { return boost::multiprecision::isinf(x); }
// auto sqrt(const precise_float_t& x) { return boost::multiprecision::sqrt(x); }
// } // namespace std

// // auto log2(const precise_float_t& x) { return log2q(x); }
// // auto abs(const precise_float_t& x) { return absq(x); }
// // auto exp2(const precise_float_t& x) { return exp2q(x); }
// // auto isnan(const precise_float_t& x) { return isnanq(x); }
// // auto isinf(const precise_float_t& x) { return isinfq(x); }
// // auto sqrt(const precise_float_t& x) { return sqrtq(x); }

// #endif
#endif
