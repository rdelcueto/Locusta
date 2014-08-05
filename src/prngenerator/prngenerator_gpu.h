#ifndef _PRNGENERATOR_GPU_H_
#define _PRNGENERATOR_GPU_H_

//#define __extern_always_inline

#include <sys/time.h>
#include "prngenerator.h"

#include <curand.h>
#include <curand_kernel.h>
#include "./cuda_common/curand_helper.h"
#include "./cuda_common/cuda_helpers.h"

namespace locusta {

    template <typename TFloat>
        class prngenerator_gpu : public prngenerator<TFloat> {
    private:

        prngenerator_gpu();
        virtual TFloat _generate();

    public:

        prngenerator_gpu(const uint32_t num_generators);

        virtual ~prngenerator_gpu();

        virtual void _initialize_engines(uint64_t seed);

        virtual void _generate(const uint32_t n, TFloat * output);

        virtual curandState * get_device_generator_states() const;

    protected:

        using prngenerator<TFloat>::_NUM_GENERATORS;
        curandGenerator_t _bulk_prng_engine;
        curandState *_device_prng_engines;
        uint32_t _num_device_states;

    };

}

#include "prngenerator_gpu_impl.h"

#endif /* _PRNGENERATOR_GPU_H_ */
