#include "benchmarks_gpu.h"
#include "./evaluator/bound_mapping_gpu.h"

namespace locusta {

    /// GPU Kernels Shared Memory Pointer.
    extern __shared__ int evaluator_shared_memory[];
    const uint32_t REPETITIONS = 1e2;
  
    template <typename TFloat>
    __global__
    void benchmark_gpu_func_1_kernel
    (const TFloat * const UPPER_BOUNDS,
     const TFloat * const LOWER_BOUNDS,
     const uint32_t NUM_ISLES,
     const uint32_t NUM_AGENTS,
     const uint32_t NUM_DIMENSIONS,
     const uint32_t bound_mapping_method,
     const bool f_negate,
     const TFloat * const agents_data,
     TFloat * const agents_fitness)
    {
        const uint32_t gene_base_idx = blockIdx.x * NUM_DIMENSIONS * NUM_AGENTS + threadIdx.x;

        TFloat reduction_sum;
        for(uint32_t r = 0; r < REPETITIONS; ++r)
        {
            reduction_sum = 0;
            for(uint32_t i = 0; i < NUM_DIMENSIONS; ++i)
            {
                TFloat x = agents_data[gene_base_idx + i * NUM_AGENTS];

                bound_mapping(bound_mapping_method,
                              UPPER_BOUNDS[i],
                              LOWER_BOUNDS[i],
                              x);
        
                reduction_sum += x * x;
            }
        }

        const uint32_t fitness_idx = blockIdx.x * NUM_AGENTS + threadIdx.x;
        agents_fitness[fitness_idx] = f_negate ?
            -reduction_sum :
            reduction_sum;
    }

    template <typename TFloat>
    void benchmark_gpu_func_1
    (const TFloat * const UPPER_BOUNDS,
     const TFloat * const LOWER_BOUNDS,
     const uint32_t NUM_ISLES,
     const uint32_t NUM_AGENTS,
     const uint32_t NUM_DIMENSIONS,
     const uint32_t bound_mapping_method,
     const bool f_negate,
     const TFloat * const agents_data,
     TFloat * const agents_fitness)
    {
        benchmark_gpu_func_1_kernel
            <<<NUM_ISLES, NUM_AGENTS>>>
            (UPPER_BOUNDS,
             LOWER_BOUNDS,
             NUM_ISLES,
             NUM_AGENTS,
             NUM_DIMENSIONS,
             bound_mapping_method,
             f_negate,
             agents_data,
             agents_fitness);
        CudaCheckError();
    }

    // Template Specialization (float)

    template
    void benchmark_gpu_func_1<float>
    (const float * const UPPER_BOUNDS,
     const float * const LOWER_BOUNDS,
     const uint32_t NUM_ISLES,
     const uint32_t NUM_AGENTS,
     const uint32_t NUM_DIMENSIONS,
     const uint32_t bound_mapping_method,
     const bool f_negate,
     const float * const agents_data,
     float * const agents_fitness);

    // Template Specialization (double)

    template
    void benchmark_gpu_func_1<double>
    (const double * const UPPER_BOUNDS,
     const double * const LOWER_BOUNDS,
     const uint32_t NUM_ISLES,
     const uint32_t NUM_AGENTS,
     const uint32_t NUM_DIMENSIONS,
     const uint32_t bound_mapping_method,
     const bool f_negate,
     const double * const agents_data,
     double * const agents_fitness);
}
