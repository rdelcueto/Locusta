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

    /// Initializes and allocates solver's runtime resources.
    virtual void setup_solver();

    /// Terminates and deallocates solver's runtime resources.
    virtual void teardown_solver();

    /// Applies solver's population transformation.
    virtual void transform() = 0;

    /// Updates best genomes records
    virtual void update_records();

    // Crop vector values, to fit within bounds.
    virtual void crop_vector(TFloat * vec);

    /// Initializes vector to uniform random values, within the solver's bounds.
    virtual void initialize_vector(TFloat * dst_vec);

    /// Prints all current genomes and their fitness.
    virtual void print_population();

    /// Prints last transformation diff.
    virtual void print_transformation_diff();

    /// Prints solver's current best found solutions and their fitness.
    virtual void print_solutions();

    using evolutionary_solver<TFloat>::_ISLES;
    using evolutionary_solver<TFloat>::_AGENTS;
    using evolutionary_solver<TFloat>::_DIMENSIONS;

    using evolutionary_solver<TFloat>::_UPPER_BOUNDS;
    using evolutionary_solver<TFloat>::_LOWER_BOUNDS;
    using evolutionary_solver<TFloat>::_VAR_RANGES;

    using evolutionary_solver<TFloat>::_population;
    using evolutionary_solver<TFloat>::_evaluator;

    using evolutionary_solver<TFloat>::_max_agent_genome;
    using evolutionary_solver<TFloat>::_min_agent_genome;
    using evolutionary_solver<TFloat>::_max_agent_fitness;
    using evolutionary_solver<TFloat>::_min_agent_fitness;
    using evolutionary_solver<TFloat>::_max_agent_idx;
    using evolutionary_solver<TFloat>::_min_agent_idx;

    using evolutionary_solver<TFloat>::_migration_step;
    using evolutionary_solver<TFloat>::_migration_size;
    using evolutionary_solver<TFloat>::_migration_selection_size;
    using evolutionary_solver<TFloat>::_migration_idxs;
    using evolutionary_solver<TFloat>::_migration_buffer;

    using evolutionary_solver<TFloat>::_bulk_prn_generator;
    using evolutionary_solver<TFloat>::_bulk_prns;
    using evolutionary_solver<TFloat>::_bulk_size;
    using evolutionary_solver<TFloat>::_prn_sets;

    using evolutionary_solver<TFloat>::_generation_count;
    using evolutionary_solver<TFloat>::_generation_target;
    using evolutionary_solver<TFloat>::_f_initialized;

  };

} // namespace locusta
#include "evolutionary_solver_cpu_impl.hpp"
#endif
