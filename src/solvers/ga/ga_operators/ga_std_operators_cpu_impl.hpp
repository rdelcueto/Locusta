#ifndef LOCUSTA_GA_STD_OPERATORS_H
#define LOCUSTA_GA_STD_OPERATORS_H

#include "ga_operators.hpp"
#include "prngenerator/prngenerator_cpu.hpp"

namespace locusta {

    template<typename TFloat>
    struct WholeCrossover : BreedFunctor<TFloat> {

        uint32_t required_prns(ga_solver_cpu<TFloat> * solver) {
            const uint32_t ISLES = solver->_ISLES;
            const uint32_t AGENTS = solver->_AGENTS;
            const uint32_t DIMENSIONS = solver->_DIMENSIONS;

            return ISLES * (AGENTS * (1 + DIMENSIONS));
        }

        void operator()(ga_solver_cpu<TFloat> * solver)
            {
                const uint32_t ISLES = solver->_ISLES;
                const uint32_t AGENTS = solver->_AGENTS;
                const uint32_t DIMENSIONS = solver->_DIMENSIONS;
                const TFloat * VAR_RANGES = solver->_VAR_RANGES;

                const TFloat DEVIATION = 0.2;

                const uint32_t RND_OFFSET = 1 + DIMENSIONS;
                const TFloat * prn_array = const_cast<TFloat *>(solver->_prn_sets[ga_solver_cpu<TFloat>::BREEDING_SET]);
                prngenerator<TFloat> * const local_generator = solver->_bulk_prn_generator;

                const TFloat CROSSOVER_RATE = solver->_crossover_rate;
                const TFloat MUTATION_RATE = solver->_mutation_rate;
                const uint32_t DIST_LIMIT = solver->_mut_dist_iterations;
                const TFloat INV_DIST_LIMIT = 1.0 / DIST_LIMIT;

                const TFloat * parent_genomes = const_cast<TFloat *>(solver->_population->_data_array);
                TFloat * offspring_genomes = solver->_population->_transformed_data_array;

                const uint32_t * couple_selection = const_cast<uint32_t *>(solver->_couples_idx_array);

#pragma omp for collapse(2)
                for(uint32_t i = 0; i < ISLES; ++i) {
                    for(uint32_t j = 0; j < AGENTS; ++j) {
                        const uint32_t ISLE_OFFSET = AGENTS * DIMENSIONS;
                        const uint32_t BASE_IDX = i * ISLE_OFFSET + j * DIMENSIONS;

                        const TFloat * agents_prns = prn_array + i * AGENTS * RND_OFFSET + j * RND_OFFSET;

                        TFloat * offspring = offspring_genomes + BASE_IDX;
                        const TFloat * parentA = parent_genomes + BASE_IDX;

                        for(uint32_t k = 0; k < DIMENSIONS; ++k) {
                            offspring[k] = parentA[k];
                        }

                        const bool CROSSOVER_FLAG = (*agents_prns) < CROSSOVER_RATE;
                        agents_prns++;

                        if(CROSSOVER_FLAG) {
                            const uint32_t COUPLE_IDX = couple_selection[i * AGENTS + j];
                            const uint32_t COUPLE_BASE_IDX = i * ISLE_OFFSET + COUPLE_IDX * DIMENSIONS;
                            const TFloat * parentB = parent_genomes + COUPLE_BASE_IDX;

                            #pragma omp simd
                            for(uint32_t k = 0; k < DIMENSIONS; ++k) {
                                offspring[k] *= 0.5;
                                offspring[k] += parentB[k] * 0.5;
                            }
                        }

                        for(uint32_t k = 0; k < DIMENSIONS; ++k) {
                            const bool GENE_MUTATE_FLAG = (*agents_prns) < MUTATION_RATE;

                            agents_prns++;
                            if(GENE_MUTATE_FLAG) {
                                const TFloat & range = VAR_RANGES[k];

                                TFloat x = 0.0;
                                for(uint32_t n = 0; n < DIST_LIMIT; ++n) {
                                    x += local_generator->_generate();
                                }

                                x *= INV_DIST_LIMIT;
                                x -= 0.5;
                                x *= DEVIATION * range;

                                offspring[k] += x;
                            }
                        }
                    }
                }

           }
    };

    template<typename TFloat>
    struct TournamentSelection : SelectionFunctor<TFloat> {

        uint32_t required_prns(ga_solver_cpu<TFloat> * solver) {
            const uint32_t ISLES = solver->_ISLES;
            const uint32_t AGENTS = solver->_AGENTS;
            const uint32_t SELECTION_SIZE = solver->_selection_size;

            return ISLES * AGENTS *
                ((AGENTS - (1 + SELECTION_SIZE)) +
                 (SELECTION_SIZE - 1));
        }

        void operator()(ga_solver_cpu<TFloat> * solver)
            {
                const uint32_t ISLES = solver->_ISLES;
                const uint32_t AGENTS = solver->_AGENTS;

                const uint32_t SELECTION_SIZE = solver->_selection_size;
                const TFloat SELECTION_P = solver->_selection_stochastic_factor;

                const TFloat * prn_array = const_cast<TFloat *>(solver->_prn_sets[ga_solver_cpu<TFloat>::SELECTION_SET]);
                const uint32_t RND_OFFSET = ((AGENTS - (1 + SELECTION_SIZE)) +
                                             (SELECTION_SIZE - 1));

                const TFloat * fitness_array = const_cast<TFloat *>(solver->_population->_fitness_array);

                uint32_t * couple_idx_array = solver->_couples_idx_array;

                #pragma omp parallel for collapse(2)
                for(uint32_t i = 0; i < ISLES; ++i) {
                    for(uint32_t j = 0; j < AGENTS; ++j) {
                        const uint32_t ISLE_OFFSET = i * AGENTS;
                        const TFloat * agents_prns = prn_array + i * AGENTS * RND_OFFSET + j * RND_OFFSET;
                        const uint32_t idx = ISLE_OFFSET + j;

                        // Resevoir Sampling
                        uint32_t candidates[SELECTION_SIZE];

                        for(uint32_t k = 0; k < (AGENTS - 1); ++k) {
                          if (k < SELECTION_SIZE) {
                            // Fill
                            candidates[k] = k < j ? k : k + 1;
                          } else {
                            uint32_t r;
                            r = (*agents_prns++) * (k + 1);
                            if (r < SELECTION_SIZE) {
                              // Replace
                              candidates[r] = k < j ? k : k + 1;
                            }
                          }
                        }

                        // Prefetch candidates fitness
                        TFloat candidates_fitness[SELECTION_SIZE];
                        for(uint32_t k = 0; k < SELECTION_SIZE; ++k) {
                            candidates_fitness[k] = fitness_array[candidates[k] + ISLE_OFFSET];
                        }

                        // Tournament
                        bool switch_flag;
                        TFloat best_fitness = candidates_fitness[0];

                        // Prng cardinality:
                        //   SELECTION_SIZE - 1
                        for(uint32_t k = 1; k < SELECTION_SIZE; ++k) {
                            const TFloat candidate = candidates_fitness[k];
                            switch_flag = (candidate > best_fitness);

                            if((SELECTION_P != 0.0f) &&
                               (SELECTION_P >= (*agents_prns))) {
                                switch_flag = !switch_flag;
                            }

                            agents_prns++; // Advance pointer

                            if(switch_flag) {
                                best_fitness = candidate;
                                candidates[0] = candidates[k];
                            }
                        }
                        couple_idx_array[idx] = candidates[0];
                    }
                }
            }
    };
}

#endif
