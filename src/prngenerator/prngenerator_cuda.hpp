#ifndef _PRNGENERATOR_CUDA_H_
#define _PRNGENERATOR_CUDA_H_

//#define __extern_always_inline

#include <sys/time.h>
#include "prngenerator.hpp"

#include <curand.h>
#include <curand_kernel.h>

namespace locusta {

  template <typename TFloat>
  struct prngenerator_cuda : prngenerator<TFloat> {

    prngenerator_cuda();
    prngenerator_cuda(uint32_t num_engines);

    virtual ~prngenerator_cuda();

    /// Initialize engines state
    virtual void _initialize_engines(uint64_t seed);

    /// Generate n pseudo random numbers into output array.
    virtual void _generate(uint32_t n, TFloat * output);

    /// Returns cuda device engine states.
    virtual curandState * get_device_generator_states() const;

    using prngenerator<TFloat>::_NUM_ENGINES;

    /// CUDA bulk prng engine.
    curandGenerator_t _dev_bulk_prng_engine;

    /// CUDA prng kernel engines.
    curandState *_dev_prng_engines;
  };

}

#include "prngenerator_cuda_impl.hpp"
#endif /* _PRNGENERATOR_CUDA_H_ */
