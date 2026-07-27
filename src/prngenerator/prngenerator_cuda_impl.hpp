#include "cuda_common/cuda_helpers.h"
#include "cuda_common/curand_helper.h"

namespace locusta {

/**
 * @brief Dispatch function for setting up cuRAND states on the GPU.
 *
 * This function dispatches the setup of cuRAND states to the appropriate CUDA
 * implementation.
 *
 * @param seed Seed value for the PRNG engines.
 * @param curand_states Pointer to the array of cuRAND states.
 * @param num_engines Number of PRNG engines to use.
 */
template<typename TFloat>
void
gpu_setup_curand_dispatch(uint64_t seed,
                          curandState* curand_states,
                          uint32_t num_engines);

/**
 * @brief Construct a new prngenerator_cuda object.
 *
 * @param num_engines Number of PRNG engines to use.
 */
template<typename TFloat>
prngenerator_cuda<TFloat>::prngenerator_cuda(uint32_t num_engines)
  : prngenerator<TFloat>::prngenerator(num_engines)
{

  CurandSafeCall(
    curandCreateGenerator(&(_dev_bulk_prng_engine), CURAND_RNG_PSEUDO_DEFAULT));

  CudaSafeCall(cudaMalloc((void**)&(_dev_prng_engines),
                          _NUM_ENGINES * sizeof(curandState)));
}

/**
 * @brief Destroy the prngenerator_cuda object.
 */
template<typename TFloat>
prngenerator_cuda<TFloat>::~prngenerator_cuda()
{
  CurandSafeCall(curandDestroyGenerator(_dev_bulk_prng_engine));
  CudaSafeCall(cudaFree(_dev_prng_engines));
  if (_dev_scratch_float != nullptr) {
    CudaSafeCall(cudaFree(_dev_scratch_float));
  }
}

/**
 * @brief Initialize the state of the PRNG engines.
 *
 * @param seed Seed value for the PRNG engines.
 */
template<typename TFloat>
void
prngenerator_cuda<TFloat>::_initialize_engines(uint64_t seed)
{
  if (seed == 0) {
    timeval curr_time;
    gettimeofday(&curr_time, NULL);
    seed = curr_time.tv_usec;
  }

  CurandSafeCall(
    curandSetPseudoRandomGeneratorSeed(_dev_bulk_prng_engine, seed));

  gpu_setup_curand_dispatch<TFloat>(seed, _dev_prng_engines, _NUM_ENGINES);
}

/**
 * @brief Generate pseudo-random numbers.
 *
 * This method generates \p n pseudo-random numbers and stores them in the \p
 * output array.
 *
 * @param n Number of pseudo-random numbers to generate.
 * @param output Output array to store the generated pseudo-random numbers.
 */
template<typename TFloat>
void
prngenerator_cuda<TFloat>::_generate(const uint32_t n, TFloat* output)
{
  if (n > _dev_scratch_capacity) {
    if (_dev_scratch_float != nullptr) {
      CudaSafeCall(cudaFree(_dev_scratch_float));
    }
    _dev_scratch_capacity = n;
    CudaSafeCall(cudaMalloc((void**)&_dev_scratch_float, _dev_scratch_capacity * sizeof(float)));
  }
  CurandSafeCall(curandGenerateUniform(_dev_bulk_prng_engine, _dev_scratch_float, n));
  gpu_cast_float_to_tfloat_dispatch(_dev_scratch_float, output, n);
}

template<>
inline void
prngenerator_cuda<float>::_generate(const uint32_t n, float* output)
{
  CurandSafeCall(curandGenerateUniform(_dev_bulk_prng_engine, output, n));
}

template<>
inline void
prngenerator_cuda<double>::_generate(const uint32_t n, double* output)
{
  CurandSafeCall(curandGenerateUniformDouble(_dev_bulk_prng_engine, output, n));
}

/**
 * @brief Returns CUDA device engine states.
 *
 * @return curandState* Pointer to the array of CUDA device engine states.
 */
template<typename TFloat>
curandState*
prngenerator_cuda<TFloat>::get_device_generator_states() const
{
  return _dev_prng_engines;
}

}
