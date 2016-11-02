#include "cuda_common/curand_helper.h"
#include "cuda_common/cuda_helpers.h"

namespace locusta {

  template<typename TFloat>
  void gpu_setup_curand_dispatch(uint64_t seed,
                                 curandState *curand_states,
                                 uint32_t num_engines);

  template<typename TFloat>
  prngenerator_cuda<TFloat>::prngenerator_cuda(uint32_t num_engines) : prngenerator<TFloat>::prngenerator(num_engines) {

    CurandSafeCall(curandCreateGenerator(&(_dev_bulk_prng_engine),
                                         CURAND_RNG_PSEUDO_DEFAULT));

    CudaSafeCall(cudaMalloc((void **) &(_dev_prng_engines),
                            _NUM_ENGINES * sizeof(curandState)));
  }

  template<typename TFloat>
  prngenerator_cuda<TFloat>::~prngenerator_cuda() {
    CurandSafeCall(curandDestroyGenerator(_dev_bulk_prng_engine));
    CudaSafeCall(cudaFree(_dev_prng_engines));
  }

  template<typename TFloat>
  void prngenerator_cuda<TFloat>::_initialize_engines(uint64_t seed) {
    if(seed == 0) {
      timeval curr_time;
      gettimeofday( &curr_time, NULL );
      seed = curr_time.tv_usec;
    }

    CurandSafeCall(curandSetPseudoRandomGeneratorSeed(_dev_bulk_prng_engine,
                                                      seed));

    gpu_setup_curand_dispatch<TFloat>(seed,
                                      _dev_prng_engines,
                                      _NUM_ENGINES);
  }

  template<typename TFloat>
  void prngenerator_cuda<TFloat>::_generate(const uint32_t n, TFloat * output) {
    // #ifdef _DEBUG
    //     __cudaCheckMemory();
    //     std::cout << "Generating " << n << " numbers." << std::endl;
    // #endif
    CurandSafeCall(curandGenerateUniform (_dev_bulk_prng_engine,
                                          output,
                                          n));
  }

  template<>
  inline void prngenerator_cuda<double>::_generate(const uint32_t n, double * output) {
    // #ifdef _DEBUG
    //     __cudaCheckMemory();
    //     std::cout << "Generating " << n << " numbers." << std::endl;
    // #endif
    CurandSafeCall(curandGenerateUniformDouble (_dev_bulk_prng_engine,
                                                output,
                                                n));
  }

  template<typename TFloat>
  curandState * prngenerator_cuda<TFloat>::get_device_generator_states() const {
    return _dev_prng_engines;
  }

}

