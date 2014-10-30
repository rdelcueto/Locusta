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
        virtual void setup_solver();

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

        /// Regenerates the bulk_prns array.
        virtual void regenerate_prns();

        // Crop vector values, to fit within bounds.
        virtual void crop_vector(TFloat * vec);

        /// Initializes vector to uniform random values, within the solver's bounds.
        virtual void initialize_vector(TFloat * dst_vec, TFloat * tmp_vec);

        /// Prints all current genomes and their fitness.
        virtual void print_population();

        /// Prints last transformation diff.
        virtual void print_transformation_diff();

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

        /// Stores the agent's genome, which has max fitness, per isle.
        TFloat * _max_agent_genome;

        /// Stores the agent's genome, which has min fitness, per isle.
        TFloat * _min_agent_genome;

        /// Stores the agent's fitness, which has max fitness, per isle.
        TFloat * _max_agent_fitness;

        /// Stores the agent's fitness, which has min fitness, per isle.
        TFloat * _min_agent_fitness;

        /// Stores the agent's index, which has max fitness, per isle.
        uint32_t * _max_agent_idx;

        /// Stores the agent's index, which has min fitness, per isle.
        uint32_t * _min_agent_idx;

        /// Defines the migration size.
        uint32_t _migration_step;

        /// Defines the migration size.
        uint32_t _migration_size;

        /// Defines the migration selection window size.
        uint32_t _migration_selection_size;

        /// Describes the migration selection indexes.
        uint32_t * _migration_idxs;

        /// Stores the temporal migration genomes to be migrated.
        TFloat * _migration_buffer;

        /// Bulk Pseudo Random Number array
        TFloat * _bulk_prns;

        /// Describes the size of the _bulk_prnumbers array.
        uint32_t _bulk_size;

        /// Describes the locations of each pseudo random number set.
        TFloat ** _prn_sets;

        /// Counter describing the solver's current generation.
        uint32_t _generation_count;

        /// Defines the solver's target generation.
        uint32_t _generation_target;

        uint8_t _f_initialized;

    };

} // namespace locusta
#include "evolutionary_solver_impl.hpp"
#endif
