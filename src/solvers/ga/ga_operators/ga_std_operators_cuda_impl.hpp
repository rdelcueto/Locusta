#ifndef LOCUSTA_GA_STD_OPERATORS_CUDA_H
#define LOCUSTA_GA_STD_OPERATORS_CUDA_H

#include "ga_operators_cuda.hpp"
#include "../../../prngenerator/prngenerator_cuda.hpp"

namespace locusta {

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
     prngenerator_cuda<TFloat> * local_generator);

    template <typename TFloat>
    void tournament_selection_dispatch
    (const uint32_t ISLES,
     const uint32_t AGENTS,
     const uint32_t SELECTION_SIZE,
     const TFloat SELECTION_P,
     const TFloat * fitness_array,
     const TFloat * prn_array,
     uint32_t * couple_idx_array,
     uint32_t * candidates_array);

    template <typename TFloat>
    struct WholeCrossoverCuda : BreedCudaFunctor<TFloat> {

        uint32_t required_prns(ga_solver_cuda<TFloat> * solver) {
            const uint32_t ISLES = solver->_ISLES;
            const uint32_t AGENTS = solver->_AGENTS;
            const uint32_t DIMENSIONS = solver->_DIMENSIONS;

            return ISLES * (AGENTS * (1 + DIMENSIONS));
        }

        void operator()(ga_solver_cuda<TFloat> * solver) {
            const uint32_t ISLES = solver->_ISLES;
            const uint32_t AGENTS = solver->_AGENTS;
            const uint32_t DIMENSIONS = solver->_DIMENSIONS;
            const TFloat * VAR_RANGES = solver->_DEV_VAR_RANGES;

            const TFloat DEVIATION = 0.2;

            const TFloat * prn_array = const_cast<TFloat *>(solver->_prn_sets[ga_solver_cuda<TFloat>::BREEDING_SET]);
            prngenerator_cuda<TFloat> * local_generator = solver->_dev_bulk_prn_generator;

            const TFloat CROSSOVER_RATE = solver->_crossover_rate;
            const TFloat MUTATION_RATE = solver->_mutation_rate;
            const uint32_t DIST_LIMIT = solver->_mut_dist_iterations;

            const TFloat * parent_genomes = const_cast<TFloat *>(solver->_dev_population->_dev_data_array);
            TFloat * offspring_genomes = solver->_dev_population->_dev_transformed_data_array;

            const uint32_t * couple_selection = const_cast<uint32_t *>(solver->_dev_couples_idx_array);

            whole_crossover_dispatch(ISLES,
                                     AGENTS,
                                     DIMENSIONS,
                                     DEVIATION,
                                     CROSSOVER_RATE,
                                     MUTATION_RATE,
                                     DIST_LIMIT,
                                     VAR_RANGES,
                                     prn_array,
                                     couple_selection,
                                     parent_genomes,
                                     offspring_genomes,
                                     local_generator);
       }
    };

    template <typename TFloat>
    struct TournamentSelectionCuda : SelectionCudaFunctor<TFloat> {

        uint32_t required_prns(ga_solver_cuda<TFloat> * solver) {
            const uint32_t ISLES = solver->_ISLES;
            const uint32_t AGENTS = solver->_AGENTS;
            const uint32_t SELECTION_SIZE = solver->_selection_size;

            return ISLES * AGENTS *
                ((AGENTS - (1 + SELECTION_SIZE)) +
                 (SELECTION_SIZE - 1));
        }

        void operator()(ga_solver_cuda<TFloat> * solver) {
            const uint32_t ISLES = solver->_ISLES;
            const uint32_t AGENTS = solver->_AGENTS;

            const uint32_t SELECTION_SIZE = solver->_selection_size;
            const TFloat SELECTION_P = solver->_selection_stochastic_factor;

            const TFloat * prn_array = const_cast<TFloat *>(solver->_prn_sets[ga_solver_cuda<TFloat>::SELECTION_SET]);

            const TFloat * fitness_array = const_cast<TFloat *>(solver->_dev_population->_dev_fitness_array);

            uint32_t * couple_idx_array = solver->_dev_couples_idx_array;
            uint32_t * candidates_array = solver->_dev_candidates_array;

            tournament_selection_dispatch(ISLES,
                                          AGENTS,
                                          SELECTION_SIZE,
                                          SELECTION_P,
                                          fitness_array,
                                          prn_array,
                                          couple_idx_array,
                                          candidates_array);
        }
    };

}

#endif
