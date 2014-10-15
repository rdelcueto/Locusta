#ifndef LOCUSTA_EVOLUTIONARY_SOLVER_CPU_H
#define LOCUSTA_EVOLUTIONARY_SOLVER_CPU_H

#include <limits>
#include <iostream>

#include "../prngenerator/prngenerator_cpu.hpp"
#include "../evaluator/evaluator_cpu.hpp"
#include "../population/population_set_cpu.hpp"

#include "evolutionary_solver.hpp"

namespace locusta {

    /// Interface for evolutionary computing metaheuristic solver_cpus
    template<typename TFloat>
    struct evolutionary_solver_cpu : evolutionary_solver<TFloat> {

        /// Default constructor
        evolutionary_solver_cpu(population_set_cpu<TFloat> * population,
                                evaluator_cpu<TFloat> * evaluator,
                                prngenerator_cpu<TFloat> * prn_generator,
                                uint32_t generation_target,
                                TFloat * upper_bounds,
                                TFloat * lower_bounds);

        /// Default destructor
        virtual ~evolutionary_solver_cpu();

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
#include "evolutionary_solver_cpu_impl.hpp"
#endif
