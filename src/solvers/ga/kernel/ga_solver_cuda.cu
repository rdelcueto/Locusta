#include "../ga_solver_cuda.hpp"
#include "cuda_common/cuda_helpers.h"

namespace locusta {
/// GPU Kernels Shared Memory Pointer.
extern __shared__ int solver_shared_memory[];

/**
 * @brief CUDA kernel for replacing the elite population.
 *
 * This kernel replaces the elite population with the best individuals from the
 * current population.
 *
 * @param AGENTS Number of agents per isle.
 * @param DIMENSIONS Number of dimensions per agent.
 * @param min_agent_idx Device array of indices of the agents with minimum
 * fitness.
 * @param max_agent_genome Device array of genomes with maximum fitness.
 * @param max_agent_fitness Device array of maximum fitness values.
 * @param genomes Device array of genomes.
 * @param fitness Device array of fitness values.
 */
template<typename TFloat>
__global__ void
elite_population_replace_kernel(const uint32_t AGENTS,
                                const uint32_t DIMENSIONS,
                                const uint32_t* __restrict__ min_agent_idx,
                                const TFloat* __restrict__ max_agent_genome,
                                const TFloat* __restrict__ max_agent_fitness,
                                TFloat* __restrict__ genomes,
                                TFloat* __restrict__ fitness)
{

  const uint32_t i = threadIdx.x; // ISLE
  const uint32_t ISLES = blockDim.x;

  const uint32_t min_idx = min_agent_idx[i];

  const uint32_t THREAD_OFFSET = ISLES * AGENTS;
  const uint32_t BASE_IDX = min_idx + i * AGENTS;

  TFloat* min_genome = genomes + BASE_IDX;

  const TFloat* max_genome = max_agent_genome + i;

  // Replace fitness & genome.
  fitness[min_idx + i * AGENTS] = max_agent_fitness[i];
  for (uint32_t k = 0; k < DIMENSIONS; k++) {
    min_genome[k * THREAD_OFFSET] = max_genome[k * ISLES];
  }
}

/**
 * @brief Dispatch function for replacing the elite population.
 *
 * @param ISLES Number of isles in the population.
 * @param AGENTS Number of agents per isle.
 * @param DIMENSIONS Number of dimensions per agent.
 * @param min_agent_idx Array of indices of the agents with minimum fitness.
 * @param max_agent_genome Array of genomes with maximum fitness.
 * @param max_agent_fitness Array of maximum fitness values.
 * @param genomes Array of genomes.
 * @param fitness Array of fitness values.
 */
template<typename TFloat>
void
elite_population_replace_dispatch(const uint32_t ISLES,
                                  const uint32_t AGENTS,
                                  const uint32_t DIMENSIONS,
                                  const uint32_t* min_agent_idx,
                                  const TFloat* max_agent_genome,
                                  const TFloat* max_agent_fitness,
                                  TFloat* genomes,
                                  TFloat* fitness)
{
  elite_population_replace_kernel<<<1, ISLES>>>(AGENTS,
                                                DIMENSIONS,
                                                min_agent_idx,
                                                max_agent_genome,
                                                max_agent_fitness,
                                                genomes,
                                                fitness);

  CudaCheckError();
}

template void
elite_population_replace_dispatch<float>(const uint32_t ISLES,
                                         const uint32_t AGENTS,
                                         const uint32_t DIMENSIONS,
                                         const uint32_t* min_agent_idx,
                                         const float* max_agent_genome,
                                         const float* max_agent_fitness,
                                         float* genomes,
                                         float* fitness);

template void
elite_population_replace_dispatch<double>(const uint32_t ISLES,
                                          const uint32_t AGENTS,
                                          const uint32_t DIMENSIONS,
                                          const uint32_t* min_agent_idx,
                                          const double* max_agent_genome,
                                          const double* max_agent_fitness,
                                          double* genomes,
                                          double* fitness);

}
