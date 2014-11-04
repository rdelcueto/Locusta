#include "cuda_common/cuda_helpers.h"
#include "../benchmarks_cuda.hpp"

namespace locusta {

    /// GPU Kernels Shared Memory Pointer.
    extern __shared__ int evaluator_shared_memory[];
    const uint32_t REPETITIONS = 1e2;

    template <typename TFloat>
    __device__
    TFloat sphere(const uint32_t DIMENSIONS,
                  const uint32_t DIMENSION_OFFSET,
                  const TFloat * evaluation_vector) {

        TFloat reduction_sum = 0;

        for(uint32_t k = 0; k < DIMENSIONS; ++k) {
            const TFloat x = evaluation_vector[k * DIMENSION_OFFSET];
            reduction_sum += x * x;
        }

        return reduction_sum;
    }

    template <typename TFloat>
    __global__
    void benchmark_kernel
    (const uint32_t DIMENSIONS,
     const bool F_NEGATE_EVALUATION,
     const uint32_t FUNC_ID,
     const TFloat EVALUATION_BIAS,
     const TFloat * __restrict__ SHIFT_ORIGIN,
     const bool F_ROTATE,
     const TFloat * __restrict__ ROTATION_MATRIX,
     const TFloat * __restrict__ evaluation_data,
     TFloat * __restrict__ evaluation_results,
     curandState * __restrict__ local_generator) {

        const uint32_t isle = blockIdx.x;
        const uint32_t agent = threadIdx.x;

        const uint32_t ISLES = gridDim.x;
        const uint32_t AGENTS = blockDim.x;

        const uint32_t genome_base = isle * AGENTS + agent;
        const uint32_t gene_offset = ISLES * AGENTS;

        const TFloat * genome = evaluation_data + genome_base;
        TFloat result = 0;

        for(uint32_t r = 0; r < REPETITIONS; ++r) {
            switch (FUNC_ID) {
            default:
                result = sphere(DIMENSIONS,
                                gene_offset,
                                genome);
               break;
            }
        }

        const uint32_t fitness_idx = isle * AGENTS + agent;
        evaluation_results[fitness_idx] = F_NEGATE_EVALUATION ?
            -result : result;
    }

    template <typename TFloat>
    void benchmark_dispatch
    (const uint32_t ISLES,
     const uint32_t AGENTS,
     const uint32_t DIMENSIONS,
     const bool F_NEGATE_EVALUATION,
     const uint32_t FUNC_ID,
     const TFloat EVALUATION_BIAS,
     const TFloat * SHIFT_ORIGIN,
     const bool F_ROTATE,
     const TFloat * ROTATION_MATRIX,
     const TFloat * evaluation_data,
     TFloat * evaluation_results,
     prngenerator_cuda<TFloat> * local_generator) {


        curandState * device_generators = local_generator->get_device_generator_states();

        benchmark_kernel
            <<<ISLES, AGENTS>>>
            (DIMENSIONS,
             F_NEGATE_EVALUATION,
             FUNC_ID,
             EVALUATION_BIAS,
             SHIFT_ORIGIN,
             F_ROTATE,
             ROTATION_MATRIX,
             evaluation_data,
             evaluation_results,
             device_generators);

        CudaCheckError();
    }

    // Template Specialization (float)
    template
    void benchmark_dispatch<float>
    (const uint32_t ISLES,
     const uint32_t AGENTS,
     const uint32_t DIMENSIONS,
     const bool F_NEGATE_EVALUATION,
     const uint32_t FUNC_ID,
     const float EVALUATION_BIAS,
     const float * SHIFT_ORIGIN,
     const bool F_ROTATE,
     const float * ROTATION_MATRIX,
     const float * evaluation_data,
     float * evaluation_results,
     prngenerator_cuda<float> * local_generator);

    // Template Specialization (double)
    template
    void benchmark_dispatch<double>
    (const uint32_t ISLES,
     const uint32_t AGENTS,
     const uint32_t DIMENSIONS,
     const bool F_NEGATE_EVALUATION,
     const uint32_t FUNC_ID,
     const double EVALUATION_BIAS,
     const double * SHIFT_ORIGIN,
     const bool F_ROTATE,
     const double * ROTATION_MATRIX,
     const double * evaluation_data,
     double * evaluation_results,
     prngenerator_cuda<double> * local_generator);

}
