#include "../de_std_operators_cuda_impl.hpp"
#include "cuda_common/cuda_helpers.h"

namespace locusta {

/// GPU Kernels Shared Memory Pointer.
extern __shared__ int de_operators_shared_memory[];

/**
 * @brief CUDA device function for performing gene crossover in differential
 * evolution.
 *
 * @param difference_a_vector First difference vector.
 * @param difference_b_vector Second difference vector.
 * @param base_vector Base vector.
 * @param DIFFERENTIAL_SCALE_FACTOR Differential scale factor.
 * @return Trial gene.
 */
template<typename TFloat>
__device__ __forceinline__ TFloat
CrossOverGene(const TFloat difference_a_vector,
              const TFloat difference_b_vector,
              const TFloat base_vector,
              const TFloat DIFFERENTIAL_SCALE_FACTOR)
{
  TFloat trial_gene = difference_a_vector;
  trial_gene -= difference_b_vector;
  trial_gene *= DIFFERENTIAL_SCALE_FACTOR;
  trial_gene += base_vector;
  return trial_gene;
}

/**
 * @brief CUDA device function for cropping a gene to fit within the bounds.
 *
 * @param offspring_gene Offspring gene to crop.
 * @param lower_bound Lower bound of the gene.
 * @param upper_bound Upper bound of the gene.
 * @return Cropped gene.
 */
template<typename TFloat>
__device__ __forceinline__ TFloat
CropGene(const TFloat offspring_gene,
         const TFloat lower_bound,
         const TFloat upper_bound)
{
  TFloat cropped_gene;
  cropped_gene = offspring_gene < lower_bound ? lower_bound : offspring_gene;
  cropped_gene = offspring_gene > upper_bound ? upper_bound : offspring_gene;
  return cropped_gene;
}

/**
 * @brief CUDA kernel for the whole crossover operator in differential
 * evolution.
 *
 * This kernel performs crossover and mutation on the population of genomes.
 *
 * @param DIMENSIONS Number of dimensions per agent.
 * @param CROSSOVER_RATE Crossover rate.
 * @param DIFFERENTIAL_SCALE_FACTOR Differential scale factor.
 * @param VAR_RANGES Array of ranges for the genes.
 * @param LOWER_BOUNDS Array of lower bounds for the genes.
 * @param UPPER_BOUNDS Array of upper bounds for the genes.
 * @param prn_array Array of pseudo-random numbers.
 * @param trial_selection Array of trial selections.
 * @param current_vectors Array of current vectors.
 * @param trial_vectors Array of trial vectors.
 * @param local_generator Array of cuRAND states.
 */
template<typename TFloat>
__global__ void
de_whole_crossover_kernel(const uint32_t DIMENSIONS,
                          const TFloat CROSSOVER_RATE,
                          const TFloat DIFFERENTIAL_SCALE_FACTOR,
                          const TFloat* __restrict__ VAR_RANGES,
                          const TFloat* __restrict__ LOWER_BOUNDS,
                          const TFloat* __restrict__ UPPER_BOUNDS,
                          const TFloat* __restrict__ prn_array,
                          const uint32_t* __restrict__ trial_selection,
                          const TFloat* __restrict__ current_vectors,
                          TFloat* __restrict__ trial_vectors,
                          curandState* __restrict__ local_generator)
{
  // Use shared memory to broadcast bounds to all threads
  TFloat* shared_lower_bounds = (TFloat*)de_operators_shared_memory;
  TFloat* shared_upper_bounds = (TFloat*)&shared_lower_bounds[DIMENSIONS];

  // If Block is big enough, fill up shared memory in parallel
  if (blockDim.x >= DIMENSIONS) {
    if (threadIdx.x < DIMENSIONS) {
      shared_lower_bounds[threadIdx.x] = LOWER_BOUNDS[threadIdx.x];
      shared_upper_bounds[threadIdx.x] = UPPER_BOUNDS[threadIdx.x];
    }
  } else {
    // Fill up shared memory with one thread
    if (threadIdx.x == 0) {
      for (uint32_t k = 0; k < DIMENSIONS; ++k) {
        shared_lower_bounds[k] = LOWER_BOUNDS[k];
        shared_upper_bounds[k] = UPPER_BOUNDS[k];
      }
    }
  }
  __syncthreads();

  const uint32_t i = blockIdx.x;
  const uint32_t j = threadIdx.x;

  const uint32_t ISLES = gridDim.x;
  const uint32_t AGENTS = blockDim.x;

  const uint32_t THREAD_OFFSET = ISLES * AGENTS;
  const uint32_t BASE_IDX = i * AGENTS + j;

  curandState local_state = local_generator[BASE_IDX];

  const TFloat* agent_prns = prn_array + BASE_IDX;

  const TFloat* target_vector = current_vectors + BASE_IDX;
  TFloat* trial_vector = trial_vectors + BASE_IDX;

  const uint32_t DIFFERENCE_VECTOR_A_IDX = trial_selection[BASE_IDX];
  const uint32_t DIFFERENCE_VECTOR_B_IDX =
    trial_selection[BASE_IDX + THREAD_OFFSET];
  const uint32_t BASE_VECTOR_IDX =
    trial_selection[BASE_IDX + (THREAD_OFFSET * 2)];

  const uint32_t DIFFERENCE_A_OFFSET = i * AGENTS + DIFFERENCE_VECTOR_A_IDX;
  const uint32_t DIFFERENCE_B_OFFSET = i * AGENTS + DIFFERENCE_VECTOR_B_IDX;
  const uint32_t BASE_VECTOR_OFFSET = i * AGENTS + BASE_VECTOR_IDX;

  const TFloat* difference_a_vector = current_vectors + DIFFERENCE_A_OFFSET;
  const TFloat* difference_b_vector = current_vectors + DIFFERENCE_B_OFFSET;
  const TFloat* base_vector = current_vectors + BASE_VECTOR_OFFSET;

  const bool FORCE_PARAMETER_COPY_FLAG = (*agent_prns);
  agent_prns += THREAD_OFFSET; // Advance pointer

  for (uint32_t k = 0; k < DIMENSIONS; ++k) {
    TFloat trial_gene;
    const bool CROSSOVER_FLAG = (*agent_prns) < CROSSOVER_RATE;
    agent_prns += THREAD_OFFSET; // Advance pointer

    if ((k != FORCE_PARAMETER_COPY_FLAG && !CROSSOVER_FLAG)) {
      trial_vector[k * THREAD_OFFSET] = target_vector[k * THREAD_OFFSET];
    } else {
      trial_gene = CrossOverGene(difference_a_vector[k * THREAD_OFFSET],
                                 difference_b_vector[k * THREAD_OFFSET],
                                 base_vector[k * THREAD_OFFSET],
                                 DIFFERENTIAL_SCALE_FACTOR);

      trial_gene =
        CropGene(trial_gene, shared_lower_bounds[k], shared_upper_bounds[k]);

      trial_vector[k * THREAD_OFFSET] = trial_gene;
    }
  }
  local_generator[BASE_IDX] = local_state;
}

/**
 * @brief Dispatch function for the whole crossover operator in differential
 * evolution.
 *
 * @param ISLES Number of isles in the population.
 * @param AGENTS Number of agents per isle.
 * @param DIMENSIONS Number of dimensions per agent.
 * @param CROSSOVER_RATE Crossover rate.
 * @param DIFFERENTIAL_SCALE_FACTOR Differential scale factor.
 * @param VAR_RANGES Array of ranges for the genes.
 * @param LOWER_BOUNDS Array of lower bounds for the genes.
 * @param UPPER_BOUNDS Array of upper bounds for the genes.
 * @param prn_array Array of pseudo-random numbers.
 * @param trial_selection Array of trial selections.
 * @param current_vectors Array of current vectors.
 * @param trial_vectors Array of trial vectors.
 * @param local_generator Pseudo-random number generator.
 */
template<typename TFloat>
void
de_whole_crossover_dispatch(const uint32_t ISLES,
                            const uint32_t AGENTS,
                            const uint32_t DIMENSIONS,
                            const TFloat CROSSOVER_RATE,
                            const TFloat DIFFERENTIAL_SCALE_FACTOR,
                            const TFloat* VAR_RANGES,
                            const TFloat* LOWER_BOUNDS,
                            const TFloat* UPPER_BOUNDS,
                            const TFloat* prn_array,
                            const uint32_t* trial_selection,
                            const TFloat* current_vectors,
                            TFloat* trial_vectors,
                            prngenerator_cuda<TFloat>* local_generator)
{

  curandState* device_generators =
    local_generator->get_device_generator_states();

  de_whole_crossover_kernel<<<ISLES, AGENTS, DIMENSIONS * 2 * sizeof(TFloat)>>>(
    DIMENSIONS,
    CROSSOVER_RATE,
    DIFFERENTIAL_SCALE_FACTOR,
    VAR_RANGES,
    LOWER_BOUNDS,
    UPPER_BOUNDS,
    prn_array,
    trial_selection,
    current_vectors,
    trial_vectors,
    device_generators);

  CudaCheckError();
}

/**
 * @brief CUDA kernel for the random selection operator in differential
 * evolution.
 *
 * This kernel selects random vectors from the population.
 *
 * @param prn_array Array of pseudo-random numbers.
 * @param recombination_idx_array Array of recombination indices.
 * @param recombination_reservoir_array Array of recombination reservoirs.
 */
template void
de_whole_crossover_dispatch<float>(const uint32_t ISLES,
                                   const uint32_t AGENTS,
                                   const uint32_t DIMENSIONS,
                                   const float CROSSOVER_RATE,
                                   const float DIFFERENTIAL_SCALE_FACTOR,
                                   const float* VAR_RANGES,
                                   const float* LOWER_BOUNDS,
                                   const float* UPPER_BOUNDS,
                                   const float* prn_array,
                                   const uint32_t* trial_selection,
                                   const float* current_vectors,
                                   float* trial_vectors,
                                   prngenerator_cuda<float>* local_generator);

template void
de_whole_crossover_dispatch<double>(const uint32_t ISLES,
                                    const uint32_t AGENTS,
                                    const uint32_t DIMENSIONS,
                                    const double CROSSOVER_RATE,
                                    const double DIFFERENTIAL_SCALE_FACTOR,
                                    const double* VAR_RANGES,
                                    const double* LOWER_BOUNDS,
                                    const double* UPPER_BOUNDS,
                                    const double* prn_array,
                                    const uint32_t* trial_selection,
                                    const double* current_vectors,
                                    double* trial_vectors,
                                    prngenerator_cuda<double>* local_generator);

/**
 * @brief Dispatch function for the random selection operator in differential
 * evolution.
 *
 * @param ISLES Number of isles in the population.
 * @param AGENTS Number of agents per isle.
 * @param prn_array Array of pseudo-random numbers.
 * @param recombination_idx_array Array of recombination indices.
 * @param recombination_reservoir_array Array of recombination reservoirs.
 */
template<typename TFloat>
__global__ void
de_random_selection_kernel(const TFloat* __restrict__ prn_array,
                           uint32_t* __restrict__ recombination_idx_array,
                           uint32_t* __restrict__ recombination_reservoir_array)
{

  const uint32_t i = blockIdx.x;  // ISLE
  const uint32_t j = threadIdx.x; // AGENT

  const uint32_t ISLES = gridDim.x;
  const uint32_t AGENTS = blockDim.x;

  const uint32_t THREAD_OFFSET = ISLES * AGENTS;
  const uint32_t BASE_IDX = i * AGENTS + j;

  const TFloat* agent_prns = prn_array + BASE_IDX;

  uint32_t* local_candidates = recombination_reservoir_array + BASE_IDX;

  // Resevoir Sampling
  const uint32_t RANDOM_VECTORS = 3;
  const uint32_t SAMPLE_SIZE = RANDOM_VECTORS;

  for (uint32_t k = 0; k < (AGENTS - 1); ++k) {
    if (k < RANDOM_VECTORS) {
      // Fill
      local_candidates[k * THREAD_OFFSET] = k < j ? k : k + 1;
    } else {
      uint32_t r;
      r = (*agent_prns) * (k + 1);
      agent_prns += THREAD_OFFSET; // Advance pointer
      if (r < SAMPLE_SIZE) {
        // Replace
        local_candidates[r * THREAD_OFFSET] = k < j ? k : k + 1;
      }
    }
  }

  for (uint32_t k = 0; k < RANDOM_VECTORS; ++k) {
    recombination_idx_array[BASE_IDX + k * THREAD_OFFSET] =
      local_candidates[k * THREAD_OFFSET];
  }
}

template<typename TFloat>
void
de_random_selection_dispatch(const uint32_t ISLES,
                             const uint32_t AGENTS,
                             const TFloat* prn_array,
                             uint32_t* recombination_idx_array,
                             uint32_t* recombination_reservoir_array)
{
  de_random_selection_kernel<<<ISLES, AGENTS>>>(
    prn_array, recombination_idx_array, recombination_reservoir_array);

  CudaCheckError();
}

template void
de_random_selection_dispatch<float>(const uint32_t ISLES,
                                    const uint32_t AGENTS,
                                    const float* prn_array,
                                    uint32_t* recombination_idx_array,
                                    uint32_t* recombination_reservoir_array);

template void
de_random_selection_dispatch<double>(const uint32_t ISLES,
                                     const uint32_t AGENTS,
                                     const double* prn_array,
                                     uint32_t* recombination_idx_array,
                                     uint32_t* recombination_reservoir_array);
}
