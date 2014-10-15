#ifndef _PRNGENERATOR_CPU_H_
#define _PRNGENERATOR_CPU_H_

///#define __extern_always_inline

#include <omp.h>
#include <random>
#include <chrono>

#include "prngenerator.hpp"

namespace locusta {

    template <typename TFloat>
    struct prngenerator_cpu : prngenerator<TFloat> {

        typedef std::mt19937 mersenne_twister;
        typedef std::uniform_real_distribution<TFloat> uni_real_dist;

        prngenerator_cpu();
        prngenerator_cpu(uint32_t num_engines);

        virtual ~prngenerator_cpu();

        /// Initialize engines state
        virtual void _initialize_engines(uint64_t seed);

        /// Generate n pseudo random numbers into output array.
        virtual void _generate(uint32_t n, TFloat * output);

        /// Generate and return a pseudo random number.
        virtual TFloat _generate();

        using prngenerator<TFloat>::_NUM_ENGINES;
        mersenne_twister *_prng_engines;
    };
}
#include "prngenerator_cpu_impl.hpp"
#endif /* _PRNGENERATOR_CPU_H_ */
