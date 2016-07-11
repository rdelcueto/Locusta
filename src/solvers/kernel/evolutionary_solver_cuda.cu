#include "cuda_common/cuda_helpers.h"
#include "../evolutionary_solver_cuda.hpp"

namespace locusta {

  /// GPU Kernels Shared Memory Pointer.
  extern __shared__ int solver_shared_memory[];

  template<typename TFloat>
  __global__
  void update_records_kernel(const uint32_t DIMENSIONS,
                             const TFloat * __restrict__ data_array,
                             const TFloat * __restrict__ fitness_array,
                             TFloat * __restrict__ max_agent_genome,
                             TFloat * __restrict__ min_agent_genome,
                             TFloat * __restrict__ max_agent_fitness,
                             TFloat * __restrict__ min_agent_fitness) {
    const uint32_t i = blockIdx.x; // ISLE
    const uint32_t j = threadIdx.x; // AGENT

    const uint32_t ISLES = gridDim.x;
    const uint32_t AGENTS = blockDim.x;

    // GPU Shared memory
    uint32_t * max_idx_reduction = (uint32_t *) solver_shared_memory;
    uint32_t * min_idx_reduction = (uint32_t *) &max_idx_reduction[AGENTS];
    TFloat * max_eval_reduction = (TFloat *) &min_idx_reduction[AGENTS];
    TFloat * min_eval_reduction = (TFloat *) &max_eval_reduction[AGENTS];

    // Initialize per thread fitness values
    TFloat a, b;
    max_idx_reduction[j] = j;
    max_eval_reduction[j] = fitness_array[j + i * AGENTS];

    min_idx_reduction[j] = j;
    min_eval_reduction[j] = fitness_array[j + i * AGENTS];

    int reduction_idx = 1;
    const int reduction_limit = AGENTS;

    // Calculate reduction steps
    while(reduction_idx < reduction_limit)
      {
        reduction_idx <<= 1;
      }

    // Parallel reduction
    while(reduction_idx != 0)
      {
        if(j < reduction_idx &&
           j + reduction_idx < reduction_limit)
          {
            a = max_eval_reduction[j];
            b = max_eval_reduction[j + reduction_idx];
            if(b > a)
              {
                max_eval_reduction[j] = b;
                max_idx_reduction[j] = max_idx_reduction[j + reduction_idx];
              }
            if(b < a)
              {
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
        for(uint32_t k = 0; k < DIMENSIONS; k++) {
          max_agent_genome[i + ISLES * k] =
            data_array[max_isle_idx + i * AGENTS + k * AGENTS * ISLES];
        }
      }

      if (curr_min > min_eval_reduction[0]) {
        // Update isle's min record fitness.
        const uint32_t min_isle_idx = min_idx_reduction[0];
        min_agent_fitness[i] = min_eval_reduction[0];
        // Copy genome into min_agent_genome
        for(uint32_t k = 0; k < DIMENSIONS; k++) {
          min_agent_genome[i + ISLES * k] =
            data_array[min_isle_idx + i * AGENTS + k * AGENTS * ISLES];
        }
      }
    }
  }

  template<typename TFloat>
  void update_records_dispatch(const uint32_t ISLES,
                               const uint32_t AGENTS,
                               const uint32_t DIMENSIONS,
                               const TFloat * data_array,
                               const TFloat * fitness_array,
                               TFloat * max_agent_genome,
                               TFloat * min_agent_genome,
                               TFloat * max_agent_fitness,
                               TFloat * min_agent_fitness) {
    update_records_kernel
      <<<ISLES, AGENTS,
      2 * AGENTS * (sizeof(uint32_t) + sizeof(TFloat))>>>
      (DIMENSIONS,
       data_array,
       fitness_array,
       max_agent_genome,
       min_agent_genome,
       max_agent_fitness,
       min_agent_fitness);

    CudaCheckError();
  }

  template
  void update_records_dispatch<float>(const uint32_t ISLES,
                                      const uint32_t AGENTS,
                                      const uint32_t DIMENSIONS,
                                      const float * data_array,
                                      const float * fitness_array,
                                      float * max_agent_genome,
                                      float * min_agent_genome,
                                      float * max_agent_fitness,
                                      float * min_agent_fitness);

  template
  void update_records_dispatch<double>(const uint32_t ISLES,
                                       const uint32_t AGENTS,
                                       const uint32_t DIMENSIONS,
                                       const double * data_array,
                                       const double * fitness_array,
                                       double * max_agent_genome,
                                       double * min_agent_genome,
                                       double * max_agent_fitness,
                                       double * min_agent_fitness);

  template<typename TFloat>
  __global__
  void initialize_vector_kernel(const uint32_t DIMENSIONS,
                                const TFloat * __restrict__ LOWER_BOUNDS,
                                const TFloat * __restrict__ VAR_RANGES,
                                const TFloat * __restrict__ tmp_vec,
                                TFloat * __restrict__ dst_vec) {

    const uint32_t i = blockIdx.x; // Isle
    const uint32_t j = threadIdx.x; // Agent

    const uint32_t ISLES = gridDim.x;
    const uint32_t AGENTS = blockDim.x;

    const uint32_t locus_offset = i * AGENTS + j;
    for(uint32_t k = 0; k < DIMENSIONS; k++) {
      const uint32_t particle_gene_idx = k * ISLES * AGENTS + locus_offset;
      dst_vec[particle_gene_idx] = LOWER_BOUNDS[k] +
        (VAR_RANGES[k] * tmp_vec[particle_gene_idx]);
    }
  }

  template<typename TFloat>
  void initialize_vector_dispatch(const uint32_t ISLES,
                                  const uint32_t AGENTS,
                                  const uint32_t DIMENSIONS,
                                  const TFloat * LOWER_BOUNDS,
                                  const TFloat * VAR_RANGES,
                                  const TFloat * tmp_vec,
                                  TFloat * dst_vec) {
    initialize_vector_kernel
      <<<ISLES, AGENTS>>>
      (DIMENSIONS,
       LOWER_BOUNDS,
       VAR_RANGES,
       tmp_vec,
       dst_vec);
    CudaCheckError();
  }

  // Template initializer (float)
  template
  void initialize_vector_dispatch<float>(const uint32_t ISLES,
                                         const uint32_t AGENTS,
                                         const uint32_t DIMENSIONS,
                                         const float * LOWER_BOUNDS,
                                         const float * VAR_RANGES,
                                         const float * tmp_vec,
                                         float * dst_vec);

  // Template initializer (double)
  template
  void initialize_vector_dispatch<double>(const uint32_t ISLES,
                                          const uint32_t AGENTS,
                                          const uint32_t DIMENSIONS,
                                          const double * LOWER_BOUNDS,
                                          const double * VAR_RANGES,
                                          const double * tmp_vec,
                                          double * dst_vec);

  template<typename TFloat>
  __global__
  void crop_vector_kernel(const uint32_t DIMENSIONS,
                          const TFloat * __restrict__ UPPER_BOUNDS,
                          const TFloat * __restrict__ LOWER_BOUNDS,
                          TFloat * __restrict__ vec) {
    const uint32_t i = blockIdx.x; // Isle
    const uint32_t j = threadIdx.x; // Agent

    const uint32_t ISLES = gridDim.x;
    const uint32_t AGENTS = blockDim.x;

    const uint32_t locus_offset = i * AGENTS + j;

    for(uint32_t k = 0; k < DIMENSIONS; k++) {
      const uint32_t particle_gene_idx = k * ISLES * AGENTS + locus_offset;

      const TFloat low_bound = LOWER_BOUNDS[k];
      const TFloat high_bound = UPPER_BOUNDS[k];

      const TFloat curr_value = vec[particle_gene_idx];
      TFloat crop_value = curr_value;

      crop_value = crop_value < low_bound ? low_bound : crop_value;
      crop_value = crop_value > high_bound ? high_bound : crop_value;

      // Crop
      if(curr_value != crop_value) {
        vec[particle_gene_idx] = crop_value;
      }
    }
  }

  template<typename TFloat>
  void crop_vector_dispatch(const uint32_t ISLES,
                            const uint32_t AGENTS,
                            const uint32_t DIMENSIONS,
                            const TFloat * UPPER_BOUNDS,
                            const TFloat * LOWER_BOUNDS,
                            TFloat * vec) {
    crop_vector_kernel
      <<<ISLES, AGENTS>>>
      (DIMENSIONS,
       UPPER_BOUNDS,
       LOWER_BOUNDS,
       vec);
    CudaCheckError();
  }

  // Template initializer (float)
  template
  void crop_vector_dispatch<float>(const uint32_t ISLES,
                                   const uint32_t AGENTS,
                                   const uint32_t DIMENSIONS,
                                   const float * UPPER_BOUNDS,
                                   const float * LOWER_BOUNDS,
                                   float * vec);

  // Template initializer (double)
  template
  void crop_vector_dispatch<double>(const uint32_t ISLES,
                                    const uint32_t AGENTS,
                                    const uint32_t DIMENSIONS,
                                    const double * UPPER_BOUNDS,
                                    const double * LOWER_BOUNDS,
                                    double * vec);



}
