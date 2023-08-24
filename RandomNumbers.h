// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
   \file RandomNumbers.h
   \brief Declaration of hoomd::RandomNumbers

   This header includes templated generators for various types of random numbers required used
   throughout hoomd. These work with the RandomGenerator generator that wraps random123's Philox4x32
   RNG with an API that handles streams of random numbers originated from a seed.
 */

#ifndef HOOMD_RANDOM_NUMBERS_H_
#define HOOMD_RANDOM_NUMBERS_H_


#ifdef ENABLE_CUDA
// ensure that curand is included before random123. This avoids multiple defiintion issues
// unfortunately, at the cost of random123 using the coefficients provided by curand
// for now, they are the same
#include <curand_kernel.h>
#endif

#include <math.h>
#include <Random123/philox.h>
#include <limits>
#include <type_traits>

namespace r123
    {
using std::make_signed;
using std::make_unsigned;

#if defined(__HIPCC__) || defined(_LIBCPP_HAS_NO_CONSTEXPR)

// Amazing! cuda thinks numeric_limits::max() is a __host__ function, so
// we can't use it in a device function.
//
// The LIBCPP_HAS_NO_CONSTEXP test catches situations where the libc++
// library thinks that the compiler doesn't support constexpr, but we
// think it does.  As a consequence, the library declares
// numeric_limits::max without constexpr.  This workaround should only
// affect a narrow range of compiler/library pairings.
//
// In both cases, we find max() by computing ~(unsigned)0 right-shifted
// by is_signed.
template<typename T> R123_CONSTEXPR R123_STATIC_INLINE R123_CUDA_DEVICE T maxTvalue()
    {
    typedef typename make_unsigned<T>::type uT;
    return (~uT(0)) >> std::numeric_limits<T>::is_signed;
    }
#else
template<typename T> R123_CONSTEXPR R123_STATIC_INLINE T maxTvalue()
    {
    return std::numeric_limits<T>::max();
    }
#endif

// u01: Input is a W-bit integer (signed or unsigned).  It is cast to
//   a W-bit unsigned integer, multiplied by Ftype(2^-W) and added to
//   Ftype(2^(-W-1)).  A good compiler should optimize it down to an
//   int-to-float conversion followed by a multiply and an add, which
//   might be fused, depending on the architecture.
//
//  If the input is a uniformly distributed integer, then the
//  result is a uniformly distributed floating point number in [0, 1].
//  The result is never exactly 0.0.
//  The smallest value returned is 2^-W.
//  Let M be the number of mantissa bits in Ftype.
//  If W>M  then the largest value retured is 1.0.
//  If W<=M then the largest value returned is the largest Ftype less than 1.0.
template<typename Ftype, typename Itype> R123_CUDA_DEVICE R123_STATIC_INLINE Ftype u01(Itype in)
    {
    typedef typename make_unsigned<Itype>::type Utype;
    R123_CONSTEXPR Ftype factor = Ftype(1.) / (Ftype(maxTvalue<Utype>()) + Ftype(1.));
    R123_CONSTEXPR Ftype halffactor = Ftype(0.5) * factor;
#if R123_UNIFORM_FLOAT_STORE
    volatile Ftype x = Utype(in) * factor;
    return x + halffactor;
#else
    return Ftype(Utype(in)) * factor + halffactor;
#endif
    }

// uneg11: Input is a W-bit integer (signed or unsigned).  It is cast
//    to a W-bit signed integer, multiplied by Ftype(2^-(W-1)) and
//    then added to Ftype(2^(-W-2)).  A good compiler should optimize
//    it down to an int-to-float conversion followed by a multiply and
//    an add, which might be fused, depending on the architecture.
//
//  If the input is a uniformly distributed integer, then the
//  output is a uniformly distributed floating point number in [-1, 1].
//  The result is never exactly 0.0.
//  The smallest absolute value returned is 2^-(W-1)
//  Let M be the number of mantissa bits in Ftype.
//  If W>M  then the largest value retured is 1.0 and the smallest is -1.0.
//  If W<=M then the largest value returned is the largest Ftype less than 1.0
//    and the smallest value returned is the smallest Ftype greater than -1.0.
template<typename Ftype, typename Itype> R123_CUDA_DEVICE R123_STATIC_INLINE Ftype uneg11(Itype in)
    {
    typedef typename make_signed<Itype>::type Stype;
    R123_CONSTEXPR Ftype factor = Ftype(1.) / (Ftype(maxTvalue<Stype>()) + Ftype(1.));
    R123_CONSTEXPR Ftype halffactor = Ftype(0.5) * factor;
#if R123_UNIFORM_FLOAT_STORE
    volatile Ftype x = Stype(in) * factor;
    return x + halffactor;
#else
    return Ftype(Stype(in)) * factor + halffactor;
#endif
    }

// end code copied from random123 examples
} // namespace r123

#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif // __HIPCC__

namespace hoomd
    {
/** RNG seed

    RandomGenerator initializes with a 64-bit seed and a 128-bit counter. Seed and Counter provide
    interfaces for common seeding patterns used across HOOMD to prevent code duplication and help
    ensure that seeds are initialized correctly.

    Seed provides one constructor as we expect this to be used everywhere in HOOMD. The constructor
    is a function of the class id, the current timestep and the user seed.
*/
class Seed
    {
    public:
    /** Construct a Seed from a class ID, timestep and user seed.

        The seed is 8 bytes. Construct this from an 1 byte class id, 2 byte seed, and the lower
        5 bytes of the timestep.

        When multiple class instances can instantiate RandomGenerator objects, include values in the
        Counter that are unique to each instance. Otherwise the separate instances will generate
        identical sequences of random numbers.

        Code inside HOOMD should name the class ID value in RNGIdentifiers.h and ensure that each
        class has a unique id. External plugins should use values 200 or larger.

        id seed1 seed0 timestep4 | timestep3 timestep2 timestep1 timestep0
    */
    DEVICE Seed(uint8_t id, uint64_t timestep, uint16_t seed)
        {
        m_key = {{static_cast<uint32_t>(id) << 24 | static_cast<uint32_t>(seed) << 8
                      | static_cast<uint32_t>((timestep & 0x000000ff00000000) >> 32),
                  static_cast<uint32_t>(timestep & 0x00000000ffffffff)}};
        }

    /// Get the key
    DEVICE const r123::Philox4x32::key_type& getKey() const
        {
        return m_key;
        }

    private:
    r123::Philox4x32::key_type m_key;
    };

/** RNG Counter

    Counter provides a number of constructors that support a variety of seeding needs throughought
    HOOMD.
*/
class Counter
    {
    public:
    /** Default constructor.

    Constructs a 0 valued counter.

    Note: Only use the 4th argument when absolutely necessary and when you know that the resulting
    RNG stream will not need to sample more than 65536 values.
    */
    DEVICE Counter(uint32_t a = 0, uint32_t b = 0, uint32_t c = 0, uint16_t d = 0)
        : m_ctr({{static_cast<uint32_t>(d) << 16, c, b, a}})
        {
        }

    /// Get the counter
    DEVICE const r123::Philox4x32::ctr_type& getCounter() const
        {
        return m_ctr;
        }

    const r123::Philox4x32::ctr_type m_ctr;
    };

//! Philox random number generator
/*! random123 is a counter based random number generator. Given an input seed vector,
     it produces a random output. Outputs from one seed to the next are not correlated.
     This class implements a convenience API around random123 that allows short streams
     (less than 2**32-1) of random numbers starting from a given Seed and Counter.

     Internally, we use the philox 4x32 RNG from random123, The first two seeds map to the
     key and the remaining seeds map to the counter. One element from the counter is used
     to generate the stream of values. Constructors provide ways to conveniently initialize
     the RNG with any number of seeds or counters.

     Counter based RNGs are useful for MD simulations: See

     C.L. Phillips, J.A. Anderson, and S.C. Glotzer. "Pseudo-random number generation
     for Brownian Dynamics and Dissipative Particle Dynamics simulations on GPU devices",
     J. Comput. Phys. 230, 7191-7201 (2011).

     and

     Y. Afshar, F. Schmid, A. Pishevar, and S. Worley. "Exploiting seeding of random
     number generators for efficient domain decomposition parallelization of dissipative
     particle dynamics", Comput. Phys. Commun. 184, 1119-1128 (2013).

     for more details.
 */
class RandomGenerator
    {
    public:
    /** Construct a random generator from a Seed and a Counter

        @param seed RNG seed.
        @param counter Initial value of the RNG counter.
    */
    DEVICE inline RandomGenerator(const Seed& seed, const Counter& counter);

    /// Generate uniformly distributed 128-bit values
    DEVICE inline r123::Philox4x32::ctr_type operator()();

    /// Get the key
    DEVICE inline r123::Philox4x32::key_type getKey()
        {
        return m_key;
        }

    /// Get the counter
    DEVICE inline r123::Philox4x32::ctr_type getCounter()
        {
        return m_ctr;
        }

    private:
    r123::Philox4x32::key_type m_key; //!< RNG key
    r123::Philox4x32::ctr_type m_ctr; //!< RNG counter
    };

DEVICE inline RandomGenerator::RandomGenerator(const Seed& seed, const Counter& counter)
    {
    m_key = seed.getKey();
    m_ctr = counter.getCounter();
    }

/*! \returns A random uniform 128-bit unsigned integer.

    \post The state of the generator is advanced one step.
 */
DEVICE inline r123::Philox4x32::ctr_type RandomGenerator::operator()()
    {
    r123::Philox4x32 rng;
    r123::Philox4x32::ctr_type u = rng(m_ctr, m_key);
    m_ctr.v[0] += 1;
    return u;
    }

namespace detail
    {
//! Generate a uniform random uint32_t
template<class RNG> DEVICE inline uint32_t generate_u32(RNG& rng)
    {
    auto u = rng();
    return u.v[0];
    }

//! Generate a uniform random uint64_t
template<class RNG> DEVICE inline uint64_t generate_u64(RNG& rng)
    {
    auto u = rng();
    return uint64_t(u.v[0]) << 32 | u.v[1];
    }

//! Generate two uniform random uint64_t
/*! \param out1 [out] A random uniform 64-bit unsigned integer.
    \param out2 [out] A random uniform 64-bit unsigned integer.
 */
template<class RNG> DEVICE inline void generate_2u64(uint64_t& out1, uint64_t& out2, RNG& rng)
    {
    auto u = rng();
    out1 = uint64_t(u.v[0]) << 32 | u.v[1];
    out2 = uint64_t(u.v[2]) << 32 | u.v[3];
    }

//! Generate a random value in [2**(-65), 1]
/*!
    \returns A random uniform float in [2**(-65), 1]

    \post The state of the generator is advanced one step.
 */
template<class Real, class RNG> DEVICE inline Real generate_canonical(RNG& rng)
    {
    return r123::u01<Real>(generate_u64(rng));
    }
    } // namespace detail

//! Generate a uniform random value in [a,b]
/*! For all practical purposes, the range returned by this function is [a,b]. This is due to round
   off error: e.g. for a=1.0, 1.0+2**(-65) == 1.0. For small values of a, the range may become
   (a,b]. It depends on the round off that occurs in a + (b-a)*u, where u is in the range [2**(-65),
   1].
*/
template<typename Real> class UniformDistribution
    {
    public:
    //! Constructor
    /*! \param _a Left end point of the interval
        \param _b Right end point of the interval
    */
    DEVICE explicit UniformDistribution(Real _a = Real(0.0), Real _b = Real(1.0))
        : a(_a), width(_b - _a)
        {
        }

    //! Draw a value from the distribution
    /*! \param rng Random number generator
        \returns uniform random value in [a,b]
    */
    template<typename RNG> DEVICE inline Real operator()(RNG& rng)
        {
        return a + width * detail::generate_canonical<Real>(rng);
        }

    private:
    const Real a;     //!< Left end point of the interval
    const Real width; //!< Width of the interval
    };

}
#undef DEVICE
#endif // #define HOOMD_RANDOM_NUMBERS_H_