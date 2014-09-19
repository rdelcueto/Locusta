#ifndef LOCUSTA_GA_SOLVER_H
#define LOCUSTA_GA_SOLVER_H

#include "../evolutionary_solver.hpp"
#include "../../prngenerator/prngenerator.hpp"
#include "ga_operators.hpp"

namespace locusta {

  enum ga_prn_offsets
    {
      /// TODO: REMOVE MIGRATION
      SELECTION_PRNS_OFFSET = 0,
      BREEDING_PRNS_OFFSET = 1,
      EXTERNAL_PRNS_OFFSET = 2
    };

  ///Interface for Genetic Algorithm solvers
  template<typename TFloat>
  class ga_solver : public evolutionary_solver<TFloat> {
  public:
    ga_solver(population_set<TFloat> * population,
              evaluator<TFloat> * evaluator,
              prngenerator<TFloat> * prn_generator)
      : _migration_step(0),
        _migration_size(1),
        _migration_selection_size(2),
        _selection_size(2),
        _selection_p(0.0),
        _crossover_rate(0.9),
        _mutation_rate(0.1),
        _distribution_iterations(3.0),
        _range_extension_p(0.0),
        evolutionary_solver<TFloat>(population,
                                    evaluator,
                                    prn_generator) {}

    virtual ~ga_solver(){}

    virtual void _print_solver_config() = 0;

    virtual void _print_solver_elite() = 0;

    virtual void _print_solver_solution() = 0;

    virtual void _initialize() = 0;

    virtual void _finalize() = 0;

    /// Set Genetic Algorithm solver operators.
    virtual void _setup_operators(typename ga_operators<TFloat>::select_func selection_function,
                                  typename ga_operators<TFloat>::breed_func breeding_function,
                                  typename ga_operators<TFloat>::migrate_func migration_function) = 0;

    /// Sets up the migration operator parameters.
    virtual void _set_migration_config(uint32_t migration_step,
                                       uint32_t migration_size,
                                       uint32_t migration_selection_size) = 0;

    /// Sets up the parent selection operator parameters.
    virtual void _set_selection_config(uint32_t selection_size,
                                       TFloat selection_p) = 0;

    /// Sets up the breeding operator parameters.
    virtual void _set_breeding_config(TFloat crossover_rate,
                                      TFloat mutation_rate) = 0;

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
    virtual void _select() = 0;

    /// Runs breeding routine.
    virtual void _breed() = 0;

    /// Runs elitism storage.
    virtual void _replace_update_elite() = 0;

    /// Runs migration between isles.
    virtual void _migrate() = 0;

    /// Replaces the couple indexes.
    virtual void _set_couples_idx(uint32_t * const input_couples) = 0;

    /// Gets the couple indexes.
    virtual void _get_couples_idx(uint32_t * const output_couples) = 0;

  protected:

    /// Defines the migration size.
    uint32_t _migration_step;

    /// Defines the migration size.
    uint32_t _migration_size;

    /// Defines the migration selection window size.
    uint32_t _migration_selection_size;

    /// Defines the parent selection window size.
    uint32_t _selection_size;

    /// Defines parent selection stochastic threshold.
    TFloat _selection_p;

    /// Defines the crossover (recombination) probability for each of the genomes.
    TFloat _crossover_rate;

    /// Defines the mutation probability for each gene.
    TFloat _mutation_rate;

    /// Defines the distribution iterations when applying the mutation operator.
    TFloat _distribution_iterations;

    ///Defines the domain range percent extension.
    TFloat _range_extension_p;

    /// Stores extended domain's upper bounds.
    TFloat * _extended_upper_bounds;

    /// Stores extended domain's lower bounds.
    TFloat * _extended_lower_bounds;

    /// Describes the parent selection indexes.
    uint32_t * _coupling_idxs;

    /// Describes the migration selection indexes.
    uint32_t * _migrating_idxs;

    /// Stores the temporal migration genomes to be migrated.
    TFloat * _migration_buffer;

    /// Describes the fitness value of the isle's elite agents.
    TFloat * _elite_fitness;

    /// Describes the genomes of the isle's elite agents.
    TFloat * _elite_genomes;

    /// Describes the location of each pseudo random number set within a single isle.
    TFloat * _prn_sets[3];

    /// Describes the offset within each isle's set.
    size_t _prn_isle_offset;

    /// Function pointer to the selection method.
    typename ga_operators<TFloat>::select_func _selection_function;

    /// Function pointer to the breeding method.
    typename ga_operators<TFloat>::breed_func _breeding_function;

    /// Function pointer to the migration method.
    typename ga_operators<TFloat>::migrate_func _migration_function;

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
