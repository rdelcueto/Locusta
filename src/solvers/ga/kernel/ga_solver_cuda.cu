#include "cuda_common/cuda_helpers.h"
#include "../ga_solver_cuda.hpp"

namespace locusta {
    /// GPU Kernels Shared Memory Pointer.
    extern __shared__ int solver_shared_memory[];

    template<typename TFloat>
    __global__
    void elite_population_replace_kernel(const uint32_t AGENTS,
                                         const uint32_t DIMENSIONS,
                                         const uint32_t * __restrict__ min_agent_idx,
                                         const TFloat *  __restrict__ max_agent_genome,
                                         const TFloat *  __restrict__ max_agent_fitness,
                                         TFloat *  __restrict__ genomes,
                                         TFloat *  __restrict__ fitness) {

        const uint32_t i = threadIdx.x; // ISLE
        const uint32_t ISLES = blockDim.x;

        const uint32_t min_idx = min_agent_idx[i];

        const uint32_t THREAD_OFFSET = ISLES * AGENTS;
        const uint32_t BASE_IDX = min_idx + i * AGENTS;

        TFloat * min_genome = genomes + BASE_IDX;

        const TFloat * max_genome = max_agent_genome + i;

        // Replace fitness & genome.
        fitness[min_idx + i * AGENTS] = max_agent_fitness[i];
        for(uint32_t k = 0; k < DIMENSIONS; k++) {
            min_genome[k * THREAD_OFFSET] = max_genome[k * ISLES];
        }
    }

    template<typename TFloat>
    void elite_population_replace_dispatch(const uint32_t ISLES,
                                           const uint32_t AGENTS,
                                           const uint32_t DIMENSIONS,
                                           const uint32_t * min_agent_idx,
                                           const TFloat * max_agent_genome,
                                           const TFloat * max_agent_fitness,
                                           TFloat * genomes,
                                           TFloat * fitness) {
        elite_population_replace_kernel
            <<<1, ISLES>>>
            (AGENTS,
             DIMENSIONS,
             min_agent_idx,
             max_agent_genome,
             max_agent_fitness,
             genomes,
             fitness);

        CudaCheckError();
    }

    template
    void elite_population_replace_dispatch<float>(const uint32_t ISLES,
                                                   const uint32_t AGENTS,
                                                   const uint32_t DIMENSIONS,
                                                   const uint32_t * min_agent_idx,
                                                   const float * max_agent_genome,
                                                   const float * max_agent_fitness,
                                                   float * genomes,
                                                   float * fitness);

    template
    void elite_population_replace_dispatch<double>(const uint32_t ISLES,
                                                   const uint32_t AGENTS,
                                                   const uint32_t DIMENSIONS,
                                                   const uint32_t * min_agent_idx,
                                                   const double * max_agent_genome,
                                                   const double * max_agent_fitness,
                                                   double * genomes,
                                                   double * fitness);

}
