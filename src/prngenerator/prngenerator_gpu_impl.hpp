#include "cuda_common/curand_helper.h"
#include "cuda_common/cuda_helpers.h"

namespace locusta {

  template<typename TFloat>
  void gpu_setup_curand_dispatch(uint64_t seed,
                                 curandState *curand_states,
                                 uint32_t num_generators);

  template<typename TFloat>
  prngenerator_gpu<TFloat>::prngenerator_gpu(uint32_t num_generators) : prngenerator<TFloat>::prngenerator(num_generators)
  {

    CurandSafeCall(curandCreateGenerator(&(_bulk_prng_engine),
                                         CURAND_RNG_PSEUDO_DEFAULT));

    _num_device_states = _NUM_GENERATORS;

    CudaSafeCall(cudaMalloc((void **) &(_device_prng_engines),
                            _num_device_states * sizeof(curandState)));
  }

  template<typename TFloat>
  prngenerator_gpu<TFloat>::~prngenerator_gpu()
  {
    CurandSafeCall(curandDestroyGenerator(_bulk_prng_engine));
    CudaSafeCall(cudaFree(_device_prng_engines));
  }

  template<typename TFloat>
  void prngenerator_gpu<TFloat>::_initialize_engines(uint64_t seed)
  {
    if(seed == 0)
      {
        timeval curr_time;
        gettimeofday( &curr_time, NULL );
        seed = curr_time.tv_usec;
      }

    CurandSafeCall(curandSetPseudoRandomGeneratorSeed(_bulk_prng_engine,
                                                      seed));

    gpu_setup_curand_dispatch<TFloat>(seed,
                                      _device_prng_engines,
                                      _NUM_GENERATORS);
  }

  template<typename TFloat>
  void prngenerator_gpu<TFloat>::_generate(const uint32_t n, TFloat * output)
  {
// #ifdef _DEBUG
//     __cudaCheckMemory();
//     std::cout << "Generating " << n << " numbers." << std::endl;
// #endif
    CurandSafeCall(curandGenerateUniform (_bulk_prng_engine,
                                          output,
                                          n));
  }

  template<typename TFloat>
  TFloat prngenerator_gpu<TFloat>::_generate()
  {
    // NOT IMPLEMENTED
    return 0;
  }

  template<typename TFloat>
  curandState * prngenerator_gpu<TFloat>::get_device_generator_states() const
  {
    return _device_prng_engines;
  }
}
