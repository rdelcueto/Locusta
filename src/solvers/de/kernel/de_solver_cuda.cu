#include "../de_solver_cuda.hpp"
#include "cuda_common/cuda_helpers.h"

namespace locusta {
/// GPU Kernels Shared Memory Pointer.
extern __shared__ int solver_shared_memory[];

/**
 * @brief CUDA kernel for replacing the trial vector.
 *
 * This kernel replaces the trial vector with the best candidate solution.
 *
 * @param DIMENSIONS Number of dimensions per agent.
 * @param previous_vectors Device array of previous vectors.
 * @param previous_fitness Device array of previous fitness values.
 * @param trial_vectors Device array of trial vectors.
 * @param trial_fitness Device array of trial fitness values.
 */
template<typename TFloat>
__global__ void
trial_vector_replace_kernel(const uint32_t DIMENSIONS,
                            TFloat* __restrict__ previous_vectors,
                            const TFloat* __restrict__ previous_fitness,
                            const TFloat* __restrict__ trial_vectors,
                            TFloat* __restrict__ trial_fitness)
{

  const uint32_t i = blockIdx.x;
  const uint32_t j = threadIdx.x;

  const uint32_t ISLES = gridDim.x;
  const uint32_t AGENTS = blockDim.x;

  const uint32_t THREAD_OFFSET = ISLES * AGENTS;
  const uint32_t BASE_IDX = i * AGENTS + j;

  if (trial_fitness[BASE_IDX] > previous_fitness[BASE_IDX]) {
    const TFloat* trial_vector = trial_vectors + BASE_IDX;
    TFloat* target_vector = previous_vectors + BASE_IDX;

    for (uint32_t k = 0; k < DIMENSIONS; ++k) {
      target_vector[k * THREAD_OFFSET] = trial_vector[k * THREAD_OFFSET];
    }
  } else {
    trial_fitness[BASE_IDX] = previous_fitness[BASE_IDX];
  }
}

/**
 * @brief Dispatch function for replacing the trial vector.
 *
 * @param ISLES Number of isles in the population.
 * @param AGENTS Number of agents per isle.
 * @param DIMENSIONS Number of dimensions per agent.
 * @param previous_vectors Array of previous vectors.
 * @param previous_fitness Array of previous fitness values.
 * @param trial_vectors Array of trial vectors.
 * @param trial_fitness Array of trial fitness values.
 */
template<typename TFloat>
void
trial_vector_replace_dispatch(const uint32_t ISLES,
                              const uint32_t AGENTS,
                              const uint32_t DIMENSIONS,
                              TFloat* previous_vectors,
                              const TFloat* previous_fitness,
                              const TFloat* trial_vectors,
                              TFloat* trial_fitness)
{
  trial_vector_replace_kernel<<<ISLES, AGENTS>>>(DIMENSIONS,
                                                 previous_vectors,
                                                 previous_fitness,
                                                 trial_vectors,
                                                 trial_fitness);

  CudaCheckError();
}

template void
trial_vector_replace_dispatch<float>(const uint32_t ISLES,
                                     const uint32_t AGENTS,
                                     const uint32_t DIMENSIONS,
                                     float* previous_vectors,
                                     const float* previous_fitness,
                                     const float* trial_vectors,
                                     float* trial_fitness);

template void
trial_vector_replace_dispatch<double>(const uint32_t ISLES,
                                      const uint32_t AGENTS,
                                      const uint32_t DIMENSIONS,
                                      double* previous_vectors,
                                      const double* previous_fitness,
                                      const double* trial_vectors,
                                      double* trial_fitness);
}
