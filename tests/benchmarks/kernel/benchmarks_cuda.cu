#include "cuda_common/cuda_helpers.h"
#include "../benchmarks_cuda.hpp"

namespace locusta {

    /// GPU Kernels Shared Memory Pointer.
    extern __shared__ int evaluator_shared_memory[];
    const uint32_t REPETITIONS = 1e2;

    template <typename TFloat>
    __global__
    void benchmark_cuda_func_1_kernel(const TFloat * const UPPER_BOUNDS,
                                     const TFloat * const LOWER_BOUNDS,
                                     const uint32_t NUM_ISLES,
                                     const uint32_t NUM_AGENTS,
                                     const uint32_t NUM_DIMENSIONS,
                                     const uint32_t bound_mapping_method,
                                     const bool f_negate,
                                     const TFloat * const agents_data,
                                     TFloat * const agents_fitness)
    {
        const uint32_t isle = blockIdx.x;
        const uint32_t agent = threadIdx.x;

        const uint32_t genome_base = isle * NUM_AGENTS + agent;
        const uint32_t gene_offset = NUM_ISLES * NUM_AGENTS;

        TFloat reduction_sum;
        for(uint32_t r = 0; r < REPETITIONS; ++r)
        {
            reduction_sum = 0;
            for(uint32_t i = 0; i < NUM_DIMENSIONS; ++i)
            {
                TFloat x = agents_data[genome_base + i * gene_offset];

                const TFloat &u = UPPER_BOUNDS[i];
                const TFloat &l = LOWER_BOUNDS[i];

                //bound_mapping(bound_mapping_method, u, l, x);

                reduction_sum += x * x;
            }
        }

        const uint32_t fitness_idx = isle * NUM_AGENTS + agent;
        agents_fitness[fitness_idx] = f_negate ?
            -reduction_sum :
            reduction_sum;
    }

    template <typename TFloat>
    void benchmark_cuda_func_1
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
        benchmark_cuda_func_1_kernel
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
    void benchmark_cuda_func_1<float>
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
    void benchmark_cuda_func_1<double>
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
