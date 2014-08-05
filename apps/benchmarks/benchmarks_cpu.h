#ifndef _BENCHMARKS_CPU_H_
#define _BENCHMARKS_CPU_H_

#include "evaluator/evaluator_cpu.h"

namespace locusta {

    template<typename TFloat>
        static void benchmark_cpu_func_1(const TFloat * const UPPER_BOUNDS,
                                         const TFloat * const LOWER_BOUNDS,
                                         const uint32_t NUM_ISLES,
                                         const uint32_t NUM_AGENTS,
                                         const uint32_t NUM_DIMENSIONS,
                                         const uint32_t bound_mapping_method,
                                         const bool f_negate,
                                         const TFloat * const agents_data,
                                         TFloat * const agents_fitness);

} // namespace locusta
#include "benchmarks_cpu.cpp"
#endif /* _BENCHMARKS_CPU_H_ */
