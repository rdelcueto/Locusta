#ifndef _BENCHMARKS_GPU_H_
#define _BENCHMARKS_GPU_H_

#include "evaluator/evaluator_gpu.hpp"

namespace locusta {

  // Cuda Wrappers Forward Declarations
  template <typename TFloat>
  void benchmark_gpu_func_1(const TFloat * const UPPER_BOUNDS,
                            const TFloat * const LOWER_BOUNDS,
                            const uint32_t NUM_ISLES,
                            const uint32_t NUM_AGENTS,
                            const uint32_t NUM_DIMENSIONS,
                            const uint32_t bound_mapping_method,
                            const bool f_negate,
                            const TFloat * const agents_data,
                            TFloat * const agents_fitness);
}

#endif /* _BENCHMARKS_GPU_H_ */
