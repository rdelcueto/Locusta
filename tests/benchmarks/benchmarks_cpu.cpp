#include "evaluator/bound_mapping_cpu.hpp"

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
        for (uint32_t i = 0; i < NUM_ISLES; ++i)
        {
            for (uint32_t j = 0; j < NUM_AGENTS; ++j)
            {
                const uint32_t isle = i;
                const uint32_t agent = j;

                TFloat reduction_sum;
                for(uint32_t r = 0; r < REPETITIONS; ++r)
                {
                    reduction_sum = 0.0;
                    for (uint32_t k = 0; k < NUM_DIMENSIONS; ++k)
                    {
                        TFloat x = agents_data[isle * NUM_AGENTS * NUM_DIMENSIONS +
                                               agent * NUM_DIMENSIONS +
                                               k];

                        const TFloat &u = UPPER_BOUNDS[k];
                        const TFloat &l = LOWER_BOUNDS[k];

                        bound_mapping<TFloat>(bound_mapping_method, u, l, x);

                        reduction_sum += x * x;
                    }
                }
                agents_fitness[isle * NUM_AGENTS + agent] = f_negate ?
                    -reduction_sum :
                    reduction_sum;
            }
        }
    }
}
