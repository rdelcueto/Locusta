#include <iostream>

#include "../prngenerator_cuda.hpp"

namespace locusta {

/**
 * @brief CUDA kernel for setting up cuRAND states.
 *
 * This kernel initializes an array of cuRAND states on the GPU.
 *
 * @param seed Seed value for the PRNG engines.
 * @param curand_states Pointer to the array of cuRAND states.
 * @param num_engines Number of PRNG engines to use.
 */
template<typename TFloat>
__global__ void
gpu_setup_curand(uint64_t seed,
                 curandState* curand_states,
                 uint32_t num_engines)
{
  uint64_t id =
    blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

  while (id < num_engines) {
    curand_init(id + seed, 0, 0, &curand_states[id]);
    id += blockDim.x * gridDim.x + blockDim.y * blockDim.y;
  }
}

/**
 * @brief Dispatch function for setting up cuRAND states on the GPU.
 *
 * This function dispatches the setup of cuRAND states using the
 * corresponding engines.
 *
 * @param seed Seed value for the PRNG engines.
 * @param curand_states Pointer to the array of cuRAND states.
 * @param num_engines Number of PRNG engines to use.
 */
template<typename TFloat>
void
gpu_setup_curand_dispatch(uint64_t seed,
                          curandState* curand_states,
                          uint32_t num_engines)
{
  gpu_setup_curand<TFloat><<<32, 32>>>(seed, curand_states, num_engines);
  CudaCheckError();
}

// Explicit specialization
template void
gpu_setup_curand_dispatch<float>(uint64_t seed,
                                 curandState* curand_states,
                                 uint32_t num_engines);

template void
gpu_setup_curand_dispatch<double>(uint64_t seed,
                                  curandState* curand_states,
                                  uint32_t num_engines);

}
