#ifndef LOCUSTA_EVOLUTIONARY_SOLVER_CUDA_H
#define LOCUSTA_EVOLUTIONARY_SOLVER_CUDA_H

#include <limits>
#include <iostream>

#include "cuda_common/cuda_helpers.h"

#include "../prngenerator/prngenerator_cuda.hpp"
#include "../evaluator/evaluator_cuda.hpp"
#include "../population/population_set_cuda.hpp"

#include "evolutionary_solver.hpp"

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
        virtual void setup_solver();

        /// Terminates and deallocates solver_cuda's runtime resources.
        virtual void teardown_solver() = 0;

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

        /// Specialized device pointers
        evaluator_cuda<TFloat> * _dev_evaluator;
        population_set_cuda<TFloat> * _dev_population;
        prngenerator_cuda<TFloat> * _dev_bulk_prn_generator;

        /// Genes Variable Bounds (DEVICE COPY)
        TFloat * _DEV_UPPER_BOUNDS;
        TFloat * _DEV_LOWER_BOUNDS;

        /// Variable Ranges (DEVICE COPY)
        TFloat * _DEV_VAR_RANGES;

        /// Stores best genome found, per isle. (DEVICE COPY)
        TFloat * _dev_best_genome;

        /// Stores best genome found fitness, per isle. (DEVICE COPY)
        TFloat * _dev_best_genome_fitness;

        /// Describes the migration selection indexes. (DEVICE COPY)
        uint32_t * _dev_migration_idxs;

        /// Stores the temporal migration genomes to be migrated.
        TFloat * _dev_migration_buffer;

        /// Bulk Pseudo Random Number array
        TFloat * _dev_bulk_prnumbers;

        /// Evaluator
        using evolutionary_solver<TFloat>::_evaluator;
        /// Population Set
        using evolutionary_solver<TFloat>::_population;
        /// Bulk Pseudo Random Number Generator
        using evolutionary_solver<TFloat>::_bulk_prn_generator;

        /// Populations Configuration
        using evolutionary_solver<TFloat>::_ISLES;
        using evolutionary_solver<TFloat>::_AGENTS;
        using evolutionary_solver<TFloat>::_DIMENSIONS;

        /// Genes Variable Bounds (HOST COPY)
        using evolutionary_solver<TFloat> ::_UPPER_BOUNDS;
        using evolutionary_solver<TFloat> ::_LOWER_BOUNDS;

        /// Variable Ranges (HOST COPY)
        using evolutionary_solver<TFloat> ::_VAR_RANGES;
        /// Stores best genome found, per isle. (HOST COPY)
        using evolutionary_solver<TFloat> ::_best_genome;
        /// Stores best genome found fitness, per isle. (HOST COPY)
        using evolutionary_solver<TFloat> ::_best_genome_fitness;

        /// Defines the migration size.
        using evolutionary_solver<TFloat> ::_migration_step;

        /// Defines the migration size.
        using evolutionary_solver<TFloat> ::_migration_size;

        /// Defines the migration selection window size.
        using evolutionary_solver<TFloat> ::_migration_selection_size;

        /// Describes the migration selection indexes. (HOST COPY)
        using evolutionary_solver<TFloat> ::_migrating_idxs;

        /// Describes the size of the _bulk_prnumbers array.
        using evolutionary_solver<TFloat> ::_bulk_size;

        /// Counter describing the solver_cuda's current generation.
        using evolutionary_solver<TFloat> ::_generation_count;

        /// Defines the solver_cuda's target generation.
        using evolutionary_solver<TFloat> ::_generation_target;

        using evolutionary_solver<TFloat> ::_f_initialized;

    };

} // namespace locusta
#include "evolutionary_solver_cuda_impl.hpp"
#endif
