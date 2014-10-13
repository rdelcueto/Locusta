#ifndef LOCUSTA_EVOLUTIONARY_SOLVER_CUDA_H
#define LOCUSTA_EVOLUTIONARY_SOLVER_CUDA_H

#include <limits>

#include "../prngenerator/prngenerator_cuda.hpp"
#include "../evaluator/evaluator_cuda.hpp"
#include "../population/population_set_cuda.hpp"

#include <iostream>

namespace locusta {

    /// Interface for evolutionary computing metaheuristic solver_cudas
    template<typename TFloat>
    struct evolutionary_solver_cuda : evolutionary_solver<TFloat> {

        /// Default constructor
        evolutionary_solver_cuda(population_set_cuda<TFloat> * population,
                                 evaluator_cuda<TFloat> * evaluator,
                                 prngenerator_cuda<TFloat> * prn_generator,
                                 uint32_t generation_target,
                                 TFloat * upper_bounds,
                                 TFloat * lower_bounds);

        /// Default destructor
        virtual ~evolutionary_solver_cuda();

        /// Initializes and allocates solver_cuda's runtime resources.
        virtual void setup_solver_cuda();

        /// Terminates and deallocates solver_cuda's runtime resources.
        virtual void teardown_solver_cuda() = 0;

        /// Applies solver_cuda's population transformation.
        virtual void transform() = 0;

        /// Evolves the population through one generation step.
        virtual void advance();

        /// Runs solver_cuda until it reaches the target number of generations.
        virtual void run();

        /// Calls evaluator and assigns a fitness value to every genome.
        virtual void evaluate_genomes();

        /// Updates best genomes records
        virtual void update_records();

        /// Regenerates the bulk_prnumbers array.
        virtual void regenerate_prnumbers();

        // Crop vector values, to fit within bounds.
        virtual void crop_vector(TFloat * vec);

        /// Initializes vector to uniform random values, within the solver_cuda's bounds.
        virtual void initialize_vector(TFloat * dst_vec, TFloat * tmp_vec);

        /// Prints all current genomes and their fitness.
        virtual void print_population();

        /// Prints solver_cuda's current best found solutions and their fitness.
        virtual void print_solutions();

        /// Evaluator
        evaluator_cuda<TFloat> * const _evaluator;

        /// Bulk Pseudo Random Number Generator
        prngenerator_cuda<TFloat> * const _bulk_prn_generator;

        /// Population Set
        population_set_cuda<TFloat> * const _population;

        /// Populations Configuration
        const uint32_t _ISLES;
        const uint32_t _AGENTS;
        const uint32_t _DIMENSIONS;

        /// Genes Variable Bounds
        TFloat * _UPPER_BOUNDS;
        TFloat * _LOWER_BOUNDS;

        /// Variable Ranges
        TFloat * _VAR_RANGES;

        /// Stores best genome found, per isle.
        TFloat * _best_genome;

        /// Stores best genome found, per isle.
        TFloat * _best_genome_fitness;

        /// Defines the migration size.
        uint32_t _migration_step;

        /// Defines the migration size.
        uint32_t _migration_size;

        /// Defines the migration selection window size.
        uint32_t _migration_selection_size;

        /// Describes the migration selection indexes.
        uint32_t * _migrating_idxs;

        /// Stores the temporal migration genomes to be migrated.
        TFloat * _migration_buffer;

        /// Bulk Pseudo Random Number array
        TFloat * _bulk_prnumbers;

        /// Describes the size of the _bulk_prnumbers array.
        std::size_t _bulk_size;

        /// Counter describing the solver_cuda's current generation.
        std::size_t _generation_count;

        /// Defines the solver_cuda's target generation.
        std::size_t _generation_target;

        uint8_t _f_initialized;

    };

} // namespace locusta
#include "evolutionary_solver_cuda.cpp"
#endif
