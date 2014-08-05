#include "evaluator/bound_mapping_cpu.h"

namespace locusta {

    const uint32_t REPETITIONS = 1e2;

    template<typename TFloat>
    void benchmark_cpu_func_1(const TFloat * const UPPER_BOUNDS,
                              const TFloat * const LOWER_BOUNDS,
                              const uint32_t NUM_ISLES,
                              const uint32_t NUM_AGENTS,
                              const uint32_t NUM_DIMENSIONS,
                              const uint32_t bound_mapping_method,
                              const bool f_negate,
                              const TFloat * const agents_data,
                              TFloat * const agents_fitness)
    {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < NUM_ISLES; ++i)
        {
            for (int j = 0; j < NUM_AGENTS; ++j)
            {
                TFloat reduction_sum;
                for(uint32_t r = 0; r < REPETITIONS; ++r)
                {
                    reduction_sum = 0.0;
                    for (uint32_t k = 0; k < NUM_DIMENSIONS; ++k)
                    {

                        TFloat x = agents_data[i * NUM_AGENTS * NUM_DIMENSIONS +
                                               j * NUM_DIMENSIONS +
                                               k];
                        const TFloat &u = UPPER_BOUNDS[k];
                        const TFloat &l = LOWER_BOUNDS[k];

                        bound_mapping<TFloat>(bound_mapping_method, u, l, x);

                        reduction_sum += x * x;
                    }
                }
                agents_fitness[i*NUM_AGENTS + j] = f_negate ?
                    -reduction_sum :
                    reduction_sum;
            }
        }
    }
}
