#include "cuda_common/cuda_helpers.h"
#include "../ga_std_operators_cuda_impl.hpp"

namespace locusta {

    /// GPU Kernels Shared Memory Pointer.
    extern __shared__ int ga_operators_shared_memory[];

    template <typename TFloat>
    __global__
    void whole_crossover_kernel(const uint32_t DIMENSIONS,
                                const TFloat DEVIATION,
                                const TFloat CROSSOVER_RATE,
                                const TFloat MUTATION_RATE,
                                const uint32_t DIST_LIMIT,
                                const TFloat INV_DIST_LIMIT,
                                const TFloat * __restrict__ VAR_RANGES,
                                const TFloat * __restrict__ prn_array,
                                const uint32_t * __restrict__ couple_selection,
                                const TFloat * __restrict__ parent_genomes,
                                TFloat * __restrict__ offspring_genomes,
                                curandState * __restrict__ local_generator) {

        const uint32_t i = blockIdx.x;
        const uint32_t j = threadIdx.x;

        const uint32_t ISLES = gridDim.x;
        const uint32_t AGENTS = blockDim.x;

        const uint32_t THREAD_OFFSET = ISLES * AGENTS;
        const uint32_t BASE_IDX = i * AGENTS + j;

        curandState local_state = local_generator[BASE_IDX];

        const TFloat * agent_prns = prn_array + BASE_IDX;

        TFloat * offspring = offspring_genomes + BASE_IDX;
        const TFloat * parentA = parent_genomes + BASE_IDX;

        const bool CROSSOVER_FLAG = (*agent_prns) < CROSSOVER_RATE;
        agent_prns += THREAD_OFFSET; // Advance pointer

        for(uint32_t k = 0; k < DIMENSIONS; ++k) {
            offspring[k * THREAD_OFFSET] = parentA[k * THREAD_OFFSET];
        }

        if (CROSSOVER_FLAG) {
            const uint32_t couple_idx = couple_selection[BASE_IDX];
            const TFloat * parentB = parent_genomes + couple_idx + i * AGENTS;

            for(uint32_t k = 0; k < DIMENSIONS; ++k) {
                offspring[k * THREAD_OFFSET] *= 0.5;
                offspring[k * THREAD_OFFSET] += parentB[k * THREAD_OFFSET] * 0.5;
            }
        }

        for(uint32_t k = 0; k < DIMENSIONS; ++k) {
            const bool GENE_MUTATE_FLAG = (*agent_prns) < MUTATION_RATE;
            agent_prns += THREAD_OFFSET; // Advance pointer

            if(GENE_MUTATE_FLAG) {
                // TODO: MOVE TO SHARED MEM -> BROADCAST
                const TFloat & range = VAR_RANGES[k];

                TFloat x = 0.0;
                for(uint32_t n = 0; n < DIST_LIMIT; ++n) {
                    x += curand_uniform(&local_state);
                }

                x *= INV_DIST_LIMIT;
                x -= 0.5;
                x *= DEVIATION * range;

                offspring[k * THREAD_OFFSET] += x;
            }
        }

        local_generator[BASE_IDX] = local_state;
    }

    template <typename TFloat>
    void whole_crossover_dispatch
    (const uint32_t ISLES,
     const uint32_t AGENTS,
     const uint32_t DIMENSIONS,
     const TFloat DEVIATION,
     const TFloat CROSSOVER_RATE,
     const TFloat MUTATION_RATE,
     const uint32_t DIST_LIMIT,
     const TFloat * VAR_RANGES,
     const TFloat * prn_array,
     const uint32_t * couple_selection,
     const TFloat * parent_genomes,
     TFloat * offspring_genomes,
     prngenerator_cuda<TFloat> * local_generator) {

        curandState * device_generators = local_generator->get_device_generator_states();
        const TFloat INV_DIST_LIMIT = 1.0 / DIST_LIMIT;

        whole_crossover_kernel
            <<<ISLES, AGENTS>>>
            (DIMENSIONS,
             DEVIATION,
             CROSSOVER_RATE,
             MUTATION_RATE,
             DIST_LIMIT,
             INV_DIST_LIMIT,
             VAR_RANGES,
             prn_array,
             couple_selection,
             parent_genomes,
             offspring_genomes,
             device_generators);

        CudaCheckError();
    }

    template
    void whole_crossover_dispatch<float>
    (const uint32_t ISLES,
     const uint32_t AGENTS,
     const uint32_t DIMENSIONS,
     const float DEVIATION,
     const float CROSSOVER_RATE,
     const float MUTATION_RATE,
     const uint32_t DIST_LIMIT,
     const float * VAR_RANGES,
     const float * prn_array,
     const uint32_t * couple_selection,
     const float * parent_genomes,
     float * offspring_genomes,
     prngenerator_cuda<float> * local_generator);

    template
    void whole_crossover_dispatch<double>
    (const uint32_t ISLES,
     const uint32_t AGENTS,
     const uint32_t DIMENSIONS,
     const double DEVIATION,
     const double CROSSOVER_RATE,
     const double MUTATION_RATE,
     const uint32_t DIST_LIMIT,
     const double * VAR_RANGES,
     const double * prn_array,
     const uint32_t * couple_selection,
     const double * parent_genomes,
     double * offspring_genomes,
     prngenerator_cuda<double> * local_generator);

    template <typename TFloat>
    __global__
    void tournament_selection_kernel(const uint32_t SELECTION_SIZE,
                                     const TFloat SELECTION_P,
                                     const TFloat * __restrict__ fitness_array,
                                     const TFloat * __restrict__ prn_array,
                                     uint32_t * __restrict__ couple_idx_array,
                                     uint32_t * __restrict__ candidates_reservoir_array) {

        const uint32_t i = blockIdx.x; // ISLE
        const uint32_t j = threadIdx.x; // AGENT

        const uint32_t ISLES = gridDim.x;
        const uint32_t AGENTS = blockDim.x;

        const uint32_t THREAD_OFFSET = ISLES * AGENTS;
        const uint32_t BASE_IDX = i * AGENTS + j;

        const TFloat * agent_prns = prn_array + BASE_IDX;

        uint32_t * local_candidates = candidates_reservoir_array + BASE_IDX;

        // Resevoir Sampling
        for(uint32_t k = 0; k < (AGENTS - 1); ++k) {
          if (k < SELECTION_SIZE) {
            // Fill
            local_candidates[k * THREAD_OFFSET] = k < j ? k : k + 1;
          } else {
            uint32_t r;
            r = (*agent_prns) * (k + 1);
            agent_prns += THREAD_OFFSET; // Advance pointer
            if (r < SELECTION_SIZE) {
              // Replace
              local_candidates[r * THREAD_OFFSET] = k < j ? k : k + 1;
            }
          }
        }

        // Tournament
        bool switch_flag;

        uint32_t best_idx = *(local_candidates);
        TFloat best_fitness = fitness_array[best_idx + i * AGENTS];

        // TODO: Check prng cardinality.
        // SELECTION_SIZE - 1

        for(uint32_t k = 1; k < SELECTION_SIZE; ++k) {
            const uint32_t candidate_idx = local_candidates[k * THREAD_OFFSET];
            const TFloat candidate_fitness = fitness_array[candidate_idx + i * AGENTS];

            switch_flag = (candidate_fitness > best_fitness);

            if((SELECTION_P != 0.0f) &&
               (SELECTION_P >= (*agent_prns))) {
                switch_flag = !switch_flag;
            }

            agent_prns += THREAD_OFFSET; // Advance pointer

            if(switch_flag) {
                best_fitness = candidate_fitness;
                best_idx = candidate_idx;
            }
        }

        couple_idx_array[BASE_IDX] = best_idx;
    }


    template <typename TFloat>
    void tournament_selection_dispatch
    (const uint32_t ISLES,
     const uint32_t AGENTS,
     const uint32_t SELECTION_SIZE,
     const TFloat SELECTION_P,
     const TFloat * fitness_array,
     const TFloat * prn_array,
     uint32_t * couple_idx_array,
     uint32_t * candidates_reservoir_array) {
        tournament_selection_kernel
            <<<ISLES, AGENTS>>>
            (SELECTION_SIZE,
             SELECTION_P,
             fitness_array,
             prn_array,
             couple_idx_array,
             candidates_reservoir_array);

        CudaCheckError();
    }

    template
    void tournament_selection_dispatch<float>
    (const uint32_t ISLES,
     const uint32_t AGENTS,
     const uint32_t SELECTION_SIZE,
     const float SELECTION_P,
     const float * fitness_array,
     const float * prn_array,
     uint32_t * couple_idx_array,
     uint32_t * candidates_reservoir_array);

    template
    void tournament_selection_dispatch<double>
    (const uint32_t ISLES,
     const uint32_t AGENTS,
     const uint32_t SELECTION_SIZE,
     const double SELECTION_P,
     const double * fitness_array,
     const double * prn_array,
     uint32_t * couple_idx_array,
     uint32_t * candidates_reservoir_array);

}
