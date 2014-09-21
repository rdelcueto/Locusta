#ifndef LOCUSTA_PSO_SOLVER_H
#define LOCUSTA_PSO_SOLVER_H

#include "../evolutionary_solver.hpp"
#include "../../prngenerator/prngenerator.hpp"
#include "pso_operators.hpp"

namespace locusta {

  enum pso_prn_offsets
    {
      /// TODO: REMOVE MIGRATION
      COGNITIVE_PRNS_OFFSET = 0,
      SOCIAL_PRNS_OFFSET = 1,
      EXTERNAL_PRNS_OFFSET = 2
    };

  ///Interface for Genetic Algorithm solvers
  template<typename TFloat>
  class pso_solver : public evolutionary_solver<TFloat> {
  public:
    pso_solver(population_set<TFloat> * population,
               evaluator<TFloat> * evaluator,
               prngenerator<TFloat> * prn_generator)
      : _migration_step(0),
        _migration_size(1),
        _migration_selection_size(2),
        _max_velocity(1.0),
        _min_velocity(1.0),
        _inertia_factor(1.0),
        _cognitive_factor(1.0),
        _social_factor(1.0),
        _range_extension_p(0.0),
        evolutionary_solver<TFloat>(population,
                                    evaluator,
                                    prn_generator) {}

    virtual ~pso_solver(){}

    virtual void _print_solver_config() = 0;

    virtual void _print_solver_elite() = 0;

    virtual void _print_solver_solution() = 0;

    virtual void _initialize() = 0;

    virtual void _finalize() = 0;

    /// Set Genetic Algorithm solver operators.
    virtual void _setup_operators(typename pso_operators<TFloat>::position_func position_function,
                                  typename pso_operators<TFloat>::velocity_func velocity_function,
                                  typename pso_operators<TFloat>::migrate_func migration_function) = 0;

    /// Sets up the migration operator parameters.
    virtual void _set_migration_config(uint32_t migration_step,
                                       uint32_t migration_size,
                                       uint32_t migration_selection_size) = 0;

    /// Sets up the parent selection operator parameters.
    virtual void _set_pso_config(TFloat inertia_factor,
                                 TFloat cognitive_factor,
                                 TFloat social_factor) = 0;

    virtual void _set_velocity_limit_config(TFloat max_velocity,
                                            TFloat min_velocity) = 0;

    /// Sets the variable range extension multiplier.
    virtual void _set_range_extension(TFloat range_multiplier) = 0;

    /// Sets the generation target for the solver.
    virtual void _set_generation_target(uint32_t generation_target) = 0;

    /// Calls prngenerator and generates generation's pseudo random numbers.
    virtual void _generate_prngs() = 0;

    /// Calls evaluator and assigns a fitness value to every genome.
    virtual void _evaluate_genomes() = 0;

    /// Evolves the population through one generation step.
    virtual void _advance_generation() = 0;

    /// Runs parent selection routine.
    virtual void _update_position() = 0;

    /// Runs breeding routine.
    virtual void _update_velocity() = 0;

    /// Runs elitism storage.
    virtual void _replace_update_elite() = 0;

    /// Runs migration between isles.
    virtual void _migrate() = 0;

  protected:

    /// Defines the migration size.
    uint32_t _migration_step;

    /// Defines the migration size.
    uint32_t _migration_size;

    /// Defines the migration selection window size.
    uint32_t _migration_selection_size;

    /// Defines the maximum velocity limit.
    TFloat _max_velocity;

    /// Defines the minimum velocity limit.
    TFloat _min_velocity;

    /// Defines the inertia factor.
    TFloat _inertia_factor;

    /// Defines the weight of the particle cognitive information.
    TFloat _cognitive_factor;

    /// Defines the weight of the neighborhood social information. 
    TFloat _social_factor;

    ///Defines the domain range percent extension.
    TFloat _range_extension_p;

    /// Stores extended domain's upper bounds.
    TFloat * _extended_upper_bounds;

    /// Stores extended domain's lower bounds.
    TFloat * _extended_lower_bounds;

    /// Describes the migration selection indexes.
    uint32_t * _migrating_idxs;

    /// Stores the temporal migration genomes to be migrated.
    TFloat * _migration_buffer;

    /// Describes the fitness value of the isle's best agents.
    TFloat * _best_fitness;

    /// Describes the genomes of the isle's best agents.
    TFloat * _best_genomes;

    /// Describes the location of each pseudo random number set within a single isle.
    TFloat * _prn_sets[3];

    /// Describes the offset within each isle's set.
    size_t _prn_isle_offset;

    /// Function pointer to the position update method.
    typename pso_operators<TFloat>::position_func _update_position;

    /// Function pointer to the velocity update method.
    typename pso_operators<TFloat>::velocity_func _update_velocity;

    /// Function pointer to the migration method.
    typename pso_operators<TFloat>::migrate_func _migration_function;

    using evolutionary_solver<TFloat>::_f_initialized;

    using evolutionary_solver<TFloat>::_population;

    using evolutionary_solver<TFloat>::_evaluator;

    using evolutionary_solver<TFloat>::_bulk_prn_generator;
    using evolutionary_solver<TFloat>::_bulk_prnumbers;
    using evolutionary_solver<TFloat>::_bulk_size;

    using evolutionary_solver<TFloat>::_generation_count;
    using evolutionary_solver<TFloat>::_generation_target;

  };

} // namespace locusta
#endif

