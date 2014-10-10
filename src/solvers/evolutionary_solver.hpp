#ifndef LOCUSTA_EVOLUTIONARY_SOLVER_H
#define LOCUSTA_EVOLUTIONARY_SOLVER_H

#include <limits>

#include "../prngenerator/prngenerator.hpp"
#include "../evaluator/evaluator.hpp"
#include "../population/population_set.hpp"

#include <iostream>

namespace locusta {

    /// Interface for evolutionary computing metaheuristic solvers
    template<typename TFloat>
    struct evolutionary_solver {

        /// Default constructor
        evolutionary_solver(population_set<TFloat> * population,
                            evaluator<TFloat> * evaluator,
                            prngenerator<TFloat> * prn_generator,
                            uint32_t generation_target,
                            TFloat * upper_bounds,
                            TFloat * lower_bounds);

        /// Default destructor
        virtual ~evolutionary_solver();

        /// Initializes and allocates solver's runtime resources.
        virtual void setup_solver() = 0;

        /// Terminates and deallocates solver's runtime resources.
        virtual void teardown_solver() = 0;

        /// Applies solver's population transformation.
        virtual void transform() = 0;

        /// Evolves the population through one generation step.
        virtual void advance();

        /// Runs solver until it reaches the target number of generations.
        virtual void run();

        /// Calls evaluator and assigns a fitness value to every genome.
        virtual void evaluate_genomes();

        /// Updates best genomes records
        virtual void update_records();

        /// Regenerates the bulk_prnumbers array.
        virtual void regenerate_prnumbers();

        /// Initializes solver's population.
        virtual void initialize_population();

        /// Prints all current genomes and their fitness.
        virtual void print_population();

        /// Prints solver's current best found solutions and their fitness.
        virtual void print_solutions();

        /// Evaluator
        evaluator<TFloat> * const _evaluator;

        /// Bulk Pseudo Random Number Generator
        prngenerator<TFloat> * const _bulk_prn_generator;

        /// Population Set
        population_set<TFloat> * const _population;

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

        /// Counter describing the solver's current generation.
        std::size_t _generation_count;

        /// Defines the solver's target generation.
        std::size_t _generation_target;

        uint8_t _f_initialized;

    };

} // namespace locusta
#include "evolutionary_solver.cpp"
#endif
