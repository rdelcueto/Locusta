#include "cuda_common/cuda_helpers.h"
#include "../evolutionary_solver_cuda.hpp"

namespace locusta {

    /// GPU Kernels Shared Memory Pointer.
    extern __shared__ int solver_shared_memory[];

    template<typename TFloat>
    __global__
    void update_records_kernel(const uint32_t DIMENSIONS,
                               const TFloat * data_array,
                               const TFloat * fitness_array,
                               TFloat * best_genomes,
                               TFloat * best_genomes_fitness) {
        const uint32_t i = blockIdx.x;
        const uint32_t j = threadIdx.x;

        const uint32_t ISLES = gridDim.x;
        const uint32_t AGENTS = blockDim.x;

        // GPU Shared memory
        uint32_t * max_idx_reduction = (uint32_t *) solver_shared_memory;
        TFloat * max_eval_reduction = (TFloat *) &max_idx_reduction[AGENTS];

        // Initialize per thread fitness values
        TFloat a, b;
        max_idx_reduction[j] = j;
        max_eval_reduction[j] = fitness_array[i * AGENTS + j];

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
            }
            reduction_idx >>= 1;
            __syncthreads();
        }

        if (j == 0) {
            // Update isle's record fitness.
            best_genomes_fitness[i] = max_eval_reduction[0];
            const uint32_t best_fitness_idx = max_idx_reduction[0];
            // Copy genome into best_genomes
            for(uint32_t k = 0; k < DIMENSIONS; k++) {
                best_genomes[i + ISLES * k];
            }
        }
    }

    template<typename TFloat>
    void update_records_dispatch(const uint32_t ISLES,
                                 const uint32_t AGENTS,
                                 const uint32_t DIMENSIONS,
                                 const TFloat * data_array,
                                 const TFloat * fitness_array,
                                 TFloat * best_genomes,
                                 TFloat * best_genomes_fitness) {
        std::cout << "UPDATE RECORDS DISPATCH!" << std::endl;
        update_records_kernel
            <<<ISLES, AGENTS, AGENTS * (sizeof(uint32_t) + sizeof(TFloat))>>>
            (DIMENSIONS,
             data_array,
             fitness_array,
             best_genomes,
             best_genomes_fitness);
        CudaCheckError();
    }

    template
    void update_records_dispatch<float>(const uint32_t ISLES,
                                        const uint32_t AGENTS,
                                        const uint32_t DIMENSIONS,
                                        const float * data_array,
                                        const float * fitness_array,
                                        float * best_genomes,
                                        float * best_genomes_fitness);

    template
    void update_records_dispatch<double>(const uint32_t ISLES,
                                         const uint32_t AGENTS,
                                         const uint32_t DIMENSIONS,
                                         const double * data_array,
                                         const double * fitness_array,
                                         double * best_genomes,
                                         double * best_genomes_fitness);



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

        for(uint32_t k = 0; k < DIMENSIONS; k++) {
            const uint32_t locus_offset = k * ISLES * AGENTS;
            const uint32_t particle_gene_idx = locus_offset + i * AGENTS + j;

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

        std::cout << "INITIALIZING DISPATCH!" << std::endl;

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

        for(uint32_t k = 0; k < DIMENSIONS; k++) {

            const uint32_t locus_offset = k * ISLES * AGENTS;
            const uint32_t particle_gene_idx = locus_offset + i * AGENTS + j;
            const TFloat low_bound = LOWER_BOUNDS[k];
            const TFloat high_bound = UPPER_BOUNDS[k];

            TFloat c_value = vec[particle_gene_idx];

            c_value = c_value < low_bound ? low_bound : c_value;
            c_value = c_value > high_bound ? high_bound : c_value;

            vec[particle_gene_idx] = c_value;
        }
    }

    template<typename TFloat>
    void crop_vector_dispatch(const uint32_t ISLES,
                              const uint32_t AGENTS,
                              const uint32_t DIMENSIONS,
                              const TFloat * UPPER_BOUNDS,
                              const TFloat * LOWER_BOUNDS,
                              TFloat * vec) {

        std::cout << "CROP DISPATCH!" << std::endl;
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
