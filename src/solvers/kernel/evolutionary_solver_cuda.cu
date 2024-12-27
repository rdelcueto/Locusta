#include "../evolutionary_solver_cuda.hpp"
#include "cuda_common/cuda_helpers.h"

namespace locusta {

/// GPU Kernels Shared Memory Pointer.
extern __shared__ int solver_shared_memory[];

/**
 * @brief CUDA kernel for updating the best genomes records.
 *
 * This kernel updates the records of the best genomes found so far for each
 * isle in the population. It performs a parallel reduction to find the maximum
 * and minimum fitness values within each isle.
 *
 * @param DIMENSIONS Number of dimensions per agent.
 * @param data_array Device array of population data.
 * @param fitness_array Device array of fitness values.
 * @param max_agent_genome Device array to store the genomes with maximum
 * fitness.
 * @param min_agent_genome Device array to store the genomes with minimum
 * fitness.
 * @param max_agent_fitness Device array to store the maximum fitness values.
 * @param min_agent_fitness Device array to store the minimum fitness values.
 */
template<typename TFloat>
__global__ void
update_records_kernel(const uint32_t DIMENSIONS,
                      const TFloat* __restrict__ data_array,
                      const TFloat* __restrict__ fitness_array,
                      TFloat* __restrict__ max_agent_genome,
                      TFloat* __restrict__ min_agent_genome,
                      TFloat* __restrict__ max_agent_fitness,
                      TFloat* __restrict__ min_agent_fitness)
{
  const uint32_t i = blockIdx.x;  // ISLE
  const uint32_t j = threadIdx.x; // AGENT

  const uint32_t ISLES = gridDim.x;
  const uint32_t AGENTS = blockDim.x;

  // GPU Shared memory
  uint32_t* max_idx_reduction = (uint32_t*)solver_shared_memory;
  uint32_t* min_idx_reduction = (uint32_t*)&max_idx_reduction[AGENTS];
  TFloat* max_eval_reduction = (TFloat*)&min_idx_reduction[AGENTS];
  TFloat* min_eval_reduction = (TFloat*)&max_eval_reduction[AGENTS];

  // Initialize per thread fitness values
  TFloat a, b;
  max_idx_reduction[j] = j;
  max_eval_reduction[j] = fitness_array[j + i * AGENTS];

  min_idx_reduction[j] = j;
  min_eval_reduction[j] = fitness_array[j + i * AGENTS];

  int reduction_idx = 1;
  const int reduction_limit = AGENTS;

  // Calculate reduction steps
  while (reduction_idx < reduction_limit) {
    reduction_idx <<= 1;
  }

  // Parallel reduction
  while (reduction_idx != 0) {
    if (j < reduction_idx && j + reduction_idx < reduction_limit) {
      a = max_eval_reduction[j];
      b = max_eval_reduction[j + reduction_idx];
      if (b > a) {
        max_eval_reduction[j] = b;
        max_idx_reduction[j] = max_idx_reduction[j + reduction_idx];
      }
      if (b < a) {
        min_eval_reduction[j] = b;
        min_idx_reduction[j] = min_idx_reduction[j + reduction_idx];
      }
    }
    reduction_idx >>= 1;
    __syncthreads();
  }

  if (j == 0) {
    // Update only if it has better fitness.
    TFloat curr_max = max_agent_fitness[i];
    TFloat curr_min = min_agent_fitness[i];
    if (curr_max < max_eval_reduction[0]) {
      // Update isle's max record fitness.
      const uint32_t max_isle_idx = max_idx_reduction[0];
      max_agent_fitness[i] = max_eval_reduction[0];
      // Copy genome into max_agent_genome
      for (uint32_t k = 0; k < DIMENSIONS; k++) {
        max_agent_genome[i + ISLES * k] =
          data_array[max_isle_idx + i * AGENTS + k * AGENTS * ISLES];
      }
    }

    if (curr_min > min_eval_reduction[0]) {
      // Update isle's min record fitness.
      const uint32_t min_isle_idx = min_idx_reduction[0];
      min_agent_fitness[i] = min_eval_reduction[0];
      // Copy genome into min_agent_genome
      for (uint32_t k = 0; k < DIMENSIONS; k++) {
        min_agent_genome[i + ISLES * k] =
          data_array[min_isle_idx + i * AGENTS + k * AGENTS * ISLES];
      }
    }
  }
}

/**
 * @brief Dispatch function for updating the best genomes records.
 *
 * This function dispatches the updating of the best genomes records to the
 * appropriate CUDA implementation.
 *
 * @param ISLES Number of isles in the population.
 * @param AGENTS Number of agents per isle.
 * @param DIMENSIONS Number of dimensions per agent.
 * @param data_array Device array of population data.
 * @param fitness_array Device array of fitness values.
 * @param max_agent_genome Device array to store the genomes with maximum
 * fitness.
 * @param min_agent_genome Device array to store the genomes with minimum
 * fitness.
 * @param max_agent_fitness Device array to store the maximum fitness values.
 * @param min_agent_fitness Device array to store the minimum fitness values.
 */
template<typename TFloat>
void
update_records_dispatch(const uint32_t ISLES,
                        const uint32_t AGENTS,
                        const uint32_t DIMENSIONS,
                        const TFloat* data_array,
                        const TFloat* fitness_array,
                        TFloat* max_agent_genome,
                        TFloat* min_agent_genome,
                        TFloat* max_agent_fitness,
                        TFloat* min_agent_fitness)
{
  update_records_kernel<<<ISLES,
                          AGENTS,
                          2 * AGENTS * (sizeof(uint32_t) + sizeof(TFloat))>>>(
    DIMENSIONS,
    data_array,
    fitness_array,
    max_agent_genome,
    min_agent_genome,
    max_agent_fitness,
    min_agent_fitness);

  CudaCheckError();
}

template void
update_records_dispatch<float>(const uint32_t ISLES,
                               const uint32_t AGENTS,
                               const uint32_t DIMENSIONS,
                               const float* data_array,
                               const float* fitness_array,
                               float* max_agent_genome,
                               float* min_agent_genome,
                               float* max_agent_fitness,
                               float* min_agent_fitness);

template void
update_records_dispatch<double>(const uint32_t ISLES,
                                const uint32_t AGENTS,
                                const uint32_t DIMENSIONS,
                                const double* data_array,
                                const double* fitness_array,
                                double* max_agent_genome,
                                double* min_agent_genome,
                                double* max_agent_fitness,
                                double* min_agent_fitness);

/**
 * @brief CUDA kernel for initializing a vector with uniform random values
 * within the bounds.
 *
 * This kernel initializes a vector with uniform random values within the
 * solver's bounds.
 *
 * @param DIMENSIONS Number of dimensions per agent.
 * @param LOWER_BOUNDS Array of lower bounds for the genes.
 * @param VAR_RANGES Array of ranges for the genes.
 * @param tmp_vec Temporary vector with random values.
 * @param dst_vec Destination vector to initialize.
 */
template<typename TFloat>
__global__ void
initialize_vector_kernel(const uint32_t DIMENSIONS,
                         const TFloat* __restrict__ LOWER_BOUNDS,
                         const TFloat* __restrict__ VAR_RANGES,
                         const TFloat* __restrict__ tmp_vec,
                         TFloat* __restrict__ dst_vec)
{

  const uint32_t i = blockIdx.x;  // Isle
  const uint32_t j = threadIdx.x; // Agent

  const uint32_t ISLES = gridDim.x;
  const uint32_t AGENTS = blockDim.x;

  const uint32_t locus_offset = i * AGENTS + j;
  for (uint32_t k = 0; k < DIMENSIONS; k++) {
    const uint32_t particle_gene_idx = k * ISLES * AGENTS + locus_offset;
    dst_vec[particle_gene_idx] =
      LOWER_BOUNDS[k] + (VAR_RANGES[k] * tmp_vec[particle_gene_idx]);
  }
}

/**
 * @brief CUDA kernel for initializing a vector with uniform random values
 * within the bounds.
 *
 * This kernel initializes a vector with uniform random values within the
 * solver's bounds.
 *
 * @param DIMENSIONS Number of dimensions per agent.
 * @param LOWER_BOUNDS Array of lower bounds for the genes.
 * @param VAR_RANGES Array of ranges for the genes.
 * @param tmp_vec Temporary vector with random values.
 * @param dst_vec Destination vector to initialize.
 */
template<typename TFloat>
void
initialize_vector_dispatch(const uint32_t ISLES,
                           const uint32_t AGENTS,
                           const uint32_t DIMENSIONS,
                           const TFloat* LOWER_BOUNDS,
                           const TFloat* VAR_RANGES,
                           const TFloat* tmp_vec,
                           TFloat* dst_vec)
{
  initialize_vector_kernel<<<ISLES, AGENTS>>>(
    DIMENSIONS, LOWER_BOUNDS, VAR_RANGES, tmp_vec, dst_vec);
  CudaCheckError();
}

// Template initializer (float)
template void
initialize_vector_dispatch<float>(const uint32_t ISLES,
                                  const uint32_t AGENTS,
                                  const uint32_t DIMENSIONS,
                                  const float* LOWER_BOUNDS,
                                  const float* VAR_RANGES,
                                  const float* tmp_vec,
                                  float* dst_vec);

// Template initializer (double)
template void
initialize_vector_dispatch<double>(const uint32_t ISLES,
                                   const uint32_t AGENTS,
                                   const uint32_t DIMENSIONS,
                                   const double* LOWER_BOUNDS,
                                   const double* VAR_RANGES,
                                   const double* tmp_vec,
                                   double* dst_vec);

/**
 * @brief CUDA kernel for cropping a vector to fit within the bounds.
 *
 * This kernel crops the values of a vector to fit within the solver's bounds.
 *
 * @param DIMENSIONS Number of dimensions per agent.
 * @param UPPER_BOUNDS Array of upper bounds for the genes.
 * @param LOWER_BOUNDS Array of lower bounds for the genes.
 * @param vec Vector to crop.
 */
template<typename TFloat>
__global__ void
crop_vector_kernel(const uint32_t DIMENSIONS,
                   const TFloat* __restrict__ UPPER_BOUNDS,
                   const TFloat* __restrict__ LOWER_BOUNDS,
                   TFloat* __restrict__ vec)
{
  const uint32_t i = blockIdx.x;  // Isle
  const uint32_t j = threadIdx.x; // Agent

  const uint32_t ISLES = gridDim.x;
  const uint32_t AGENTS = blockDim.x;

  const uint32_t locus_offset = i * AGENTS + j;

  for (uint32_t k = 0; k < DIMENSIONS; k++) {
    const uint32_t particle_gene_idx = k * ISLES * AGENTS + locus_offset;

    const TFloat low_bound = LOWER_BOUNDS[k];
    const TFloat high_bound = UPPER_BOUNDS[k];

    const TFloat curr_value = vec[particle_gene_idx];
    TFloat crop_value = curr_value;

    crop_value = crop_value < low_bound ? low_bound : crop_value;
    crop_value = crop_value > high_bound ? high_bound : crop_value;

    // Crop
    if (curr_value != crop_value) {
      vec[particle_gene_idx] = crop_value;
    }
  }
}

/**
 * @brief Dispatch function for cropping a vector to fit within the bounds.
 *
 * This function dispatches the cropping of a vector to the appropriate CUDA
 * implementation.
 *
 * @param ISLES Number of isles in the population.
 * @param AGENTS Number of agents per isle.
 * @param DIMENSIONS Number of dimensions per agent.
 * @param UPPER_BOUNDS Array of upper bounds for the genes.
 * @param LOWER_BOUNDS Array of lower bounds for the genes.
 * @param vec Vector to crop.
 */
template<typename TFloat>
void
crop_vector_dispatch(const uint32_t ISLES,
                     const uint32_t AGENTS,
                     const uint32_t DIMENSIONS,
                     const TFloat* UPPER_BOUNDS,
                     const TFloat* LOWER_BOUNDS,
                     TFloat* vec)
{
  crop_vector_kernel<<<ISLES, AGENTS>>>(
    DIMENSIONS, UPPER_BOUNDS, LOWER_BOUNDS, vec);
  CudaCheckError();
}

// Template initializer (float)
template void
crop_vector_dispatch<float>(const uint32_t ISLES,
                            const uint32_t AGENTS,
                            const uint32_t DIMENSIONS,
                            const float* UPPER_BOUNDS,
                            const float* LOWER_BOUNDS,
                            float* vec);

// Template initializer (double)
template void
crop_vector_dispatch<double>(const uint32_t ISLES,
                             const uint32_t AGENTS,
                             const uint32_t DIMENSIONS,
                             const double* UPPER_BOUNDS,
                             const double* LOWER_BOUNDS,
                             double* vec);
}
