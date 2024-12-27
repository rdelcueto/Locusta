#include "../pso_solver_cuda.hpp"
#include "cuda_common/cuda_helpers.h"

namespace locusta {
/// GPU Kernels Shared Memory Pointer.
extern __shared__ int solver_shared_memory[];

/**
 * @brief CUDA kernel for resetting the velocity vector.
 *
 * This kernel resets the velocity vector of each particle to a specified value.
 *
 * @param DIMENSIONS Number of dimensions per agent.
 * @param velocity_vector Device array of velocity vectors.
 * @param reset_value Value to reset the velocity to.
 */
template<typename TFloat>
__global__ void
reset_velocity_kernel(const uint32_t DIMENSIONS,
                      TFloat* velocity_vector,
                      TFloat reset_value)
{

  const uint32_t i = blockIdx.x;  // ISLE
  const uint32_t j = threadIdx.x; // AGENT

  const uint32_t ISLES = gridDim.x;
  const uint32_t AGENTS = blockDim.x;

  const uint32_t THREAD_OFFSET = ISLES * AGENTS;
  const uint32_t BASE_IDX = j + i * AGENTS;

  TFloat* thread_vector = velocity_vector + BASE_IDX;

  // Replace fitness & genome.
  for (uint32_t k = 0; k < DIMENSIONS; k++) {
    thread_vector[k * THREAD_OFFSET] = reset_value;
  }
}

/**
 * @brief Dispatch function for resetting the velocity vector.
 *
 * @param ISLES Number of isles in the population.
 * @param AGENTS Number of agents per isle.
 * @param DIMENSIONS Number of dimensions per agent.
 * @param velocity_vector Velocity vector.
 * @param reset_value Value to reset the velocity to.
 */
template<typename TFloat>
void
reset_velocity_dispatch(const uint32_t ISLES,
                        const uint32_t AGENTS,
                        const uint32_t DIMENSIONS,
                        TFloat* velocity_vector,
                        TFloat reset_value)
{
  reset_velocity_kernel<<<ISLES, AGENTS>>>(
    DIMENSIONS, velocity_vector, reset_value);

  CudaCheckError();
}

template void
reset_velocity_dispatch<float>(const uint32_t ISLES,
                               const uint32_t AGENTS,
                               const uint32_t DIMENSIONS,
                               float* velocity_vector,
                               float reset_value);

template void
reset_velocity_dispatch<double>(const uint32_t ISLES,
                                const uint32_t AGENTS,
                                const uint32_t DIMENSIONS,
                                double* velocity_vector,
                                double reset_value);
}
