#ifndef _PRNGENERATOR_CPU_H_
#define _PRNGENERATOR_CPU_H_

///#define __extern_always_inline

#include <omp.h>
#include <random>
#include <chrono>

#include "prngenerator.h"

namespace locusta {

    template <typename TFloat>
        class prngenerator_cpu : public prngenerator<TFloat> {
    public:

        typedef std::mt19937 mersenne_twister;
        typedef std::uniform_real_distribution<TFloat> uni_real_dist;

        prngenerator_cpu();
        prngenerator_cpu(const uint32_t num_generators);

        virtual ~prngenerator_cpu();

        virtual void _initialize_engines(uint64_t seed);

        virtual void _generate(const uint32_t n, TFloat * output);

        virtual TFloat _generate();

    protected:

        using prngenerator<TFloat>::_NUM_GENERATORS;
        mersenne_twister *_prng_engines;
    };

}
#include "prngenerator_cpu.cpp"
#endif /* _PRNGENERATOR_CPU_H_ */
