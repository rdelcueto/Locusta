#include "cuda_common/cuda_helpers.h"
#include "../benchmarks_cuda.hpp"

namespace locusta {

    /// GPU Kernels Shared Memory Pointer.
    extern __shared__ int evaluator_shared_memory[];
    const uint32_t REPETITIONS = 1e2;

    template <typename TFloat>
    __global__
    void hyper_sphere_kernel
    (const uint32_t DIMENSIONS,
     const TFloat * __restrict__ UPPER_BOUNDS,
     const TFloat * __restrict__ LOWER_BOUNDS,
     BoundMapKind bound_mapping_method,
     const bool f_negate,
     const TFloat * __restrict__ data,
     TFloat * __restrict__ evaluation_array)
    {
        const uint32_t isle = blockIdx.x;
        const uint32_t agent = threadIdx.x;

        const uint32_t ISLES = gridDim.x;
        const uint32_t AGENTS = blockDim.x;

        const uint32_t genome_base = isle * AGENTS + agent;
        const uint32_t gene_offset = ISLES * AGENTS;

        TFloat reduction_sum;
        for(uint32_t r = 0; r < REPETITIONS; ++r)
        {
            reduction_sum = 0;
            for(uint32_t i = 0; i < DIMENSIONS; ++i)
            {
                TFloat x = data[genome_base + i * gene_offset];

                // const TFloat &u = UPPER_BOUNDS[i];
                // const TFloat &l = LOWER_BOUNDS[i];

                //bound_mapping(bound_mapping_method, u, l, x);

                reduction_sum += x * x;
            }
        }

        const uint32_t fitness_idx = isle * AGENTS + agent;
        evaluation_array[fitness_idx] = f_negate ?
            -reduction_sum :
            reduction_sum;
    }

    template <typename TFloat>
    void hyper_sphere_dispatch
    (const uint32_t ISLES,
     const uint32_t AGENTS,
     const uint32_t DIMENSIONS,
     const TFloat * UPPER_BOUNDS,
     const TFloat * LOWER_BOUNDS,
     BoundMapKind bound_mapping_method,
     const bool f_negate,
     const TFloat * data,
     TFloat * evaluation_array)
    {
        std::cout << "EVAL DISPATCH!" << std::endl;
        hyper_sphere_kernel
            <<<ISLES, AGENTS>>>
            (DIMENSIONS,
             UPPER_BOUNDS,
             LOWER_BOUNDS,
             bound_mapping_method,
             f_negate,
             data,
             evaluation_array);
        CudaCheckError();
    }

    // Template Specialization (float)
    template
    void hyper_sphere_dispatch<float>
    (const uint32_t ISLES,
     const uint32_t AGENTS,
     const uint32_t DIMENSIONS,
     const float * UPPER_BOUNDS,
     const float * LOWER_BOUNDS,
     BoundMapKind bound_mapping_method,
     const bool f_negate,
     const float * data,
     float * evaluation_array);

    // Template Specialization (double)
    template
    void hyper_sphere_dispatch<double>
    (const uint32_t ISLES,
     const uint32_t AGENTS,
     const uint32_t DIMENSIONS,
     const double * UPPER_BOUNDS,
     const double * LOWER_BOUNDS,
     BoundMapKind bound_mapping_method,
     const bool f_negate,
     const double * data,
     double * evaluation_array);

}
