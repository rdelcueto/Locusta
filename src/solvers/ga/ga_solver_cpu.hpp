#ifndef LOCUSTA_GA_SOLVER_CPU_H
#define LOCUSTA_GA_SOLVER_CPU_H

#include "../../prngenerator/prngenerator_cpu.hpp"
#include "../evolutionary_solver_cpu.hpp"

#include "./ga_operators/ga_operators.hpp"

namespace locusta {

/**
 * @brief CPU implementation of the genetic algorithm solver.
 *
 * This class implements the genetic algorithm solver for the CPU architecture.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct ga_solver_cpu : evolutionary_solver_cpu<TFloat>
{
  /**
   * @brief Enum defining the offsets for the pseudo-random number sets.
   */
  enum PRN_OFFSETS
  {
    SELECTION_SET = 0,
    BREEDING_SET = 1
  };

  /**
   * @brief Construct a new ga_solver_cpu object.
   *
   * @param population Population set.
   * @param evaluator Evaluator.
   * @param prn_generator Pseudo-random number generator.
   * @param generation_target Target number of generations.
   * @param upper_bounds Array of upper bounds for the genes.
   * @param lower_bounds Array of lower bounds for the genes.
   */
  ga_solver_cpu(population_set_cpu<TFloat>* population,
                evaluator_cpu<TFloat>* evaluator,
                prngenerator_cpu<TFloat>* prn_generator,
                uint64_t generation_target,
                TFloat* upper_bounds,
                TFloat* lower_bounds);

  /**
   * @brief Destroy the ga_solver_cpu object.
   */
  virtual ~ga_solver_cpu();

  /**
   * @brief Set up the solver.
   *
   * This method initializes and allocates the solver's runtime resources.
   */
  virtual void setup_solver();

  /**
   * @brief Tear down the solver.
   *
   * This method terminates and deallocates the solver's runtime resources.
   */
  virtual void teardown_solver();

  /**
   * @brief Apply the solver's population transformation.
   *
   * This method applies the solver's specific population transformation to
   * generate the next generation of candidate solutions.
   */
  virtual void transform();

  /**
   * @brief Replace the elite population.
   *
   * This method replaces the elite population with the best individuals from
   * the current population.
   */
  virtual void elite_population_replace();

  /**
   * @brief Set the genetic algorithm solver operators.
   *
   * This method sets the genetic algorithm solver operators, including the
   * breeding and selection operators.
   *
   * @param breed_functor_ptr Breeding operator.
   * @param select_functor_ptr Selection operator.
   */
  virtual void setup_operators(GaBreedFunctor<TFloat>* breed_functor_ptr,
                               GaSelectionFunctor<TFloat>* select_functor_ptr);

  /**
   * @brief Configure the solver.
   *
   * This method sets up the solver's configuration, including parameters for
   * migration, selection, crossover, and mutation.
   *
   * @param migration_step Migration step size.
   * @param migration_size Migration size.
   * @param migration_selection_size Migration selection size.
   * @param selection_size Selection size.
   * @param selection_stochastic_factor Selection stochastic factor.
   * @param crossover_rate Crossover rate.
   * @param mutation_rate Mutation rate.
   */
  virtual void solver_config(uint32_t migration_step,
                             uint32_t migration_size,
                             uint32_t migration_selection_size,
                             uint32_t selection_size,
                             TFloat selection_stochastic_factor,
                             TFloat crossover_rate,
                             TFloat mutation_rate);

  /// Population crossover + mutation operator.
  GaBreedFunctor<TFloat>* _breed_functor_ptr;

  /// Population couple selection.
  GaSelectionFunctor<TFloat>* _selection_functor_ptr;

  /// Tournament selection size.
  uint32_t _selection_size;

  /// Tournament stochastic factor
  TFloat _selection_stochastic_factor;

  /// Crossover rate.
  TFloat _crossover_rate;

  /// Mutation rate.
  TFloat _mutation_rate;

  /// Couple selection array.
  uint32_t* _couples_idx_array;

  using evolutionary_solver_cpu<TFloat>::_ISLES;
  using evolutionary_solver_cpu<TFloat>::_AGENTS;
  using evolutionary_solver_cpu<TFloat>::_DIMENSIONS;

  using evolutionary_solver_cpu<TFloat>::_UPPER_BOUNDS;
  using evolutionary_solver_cpu<TFloat>::_LOWER_BOUNDS;
  using evolutionary_solver_cpu<TFloat>::_VAR_RANGES;

  using evolutionary_solver_cpu<TFloat>::_population;
  using evolutionary_solver_cpu<TFloat>::_evaluator;

  using evolutionary_solver_cpu<TFloat>::_max_agent_genome;
  using evolutionary_solver_cpu<TFloat>::_max_agent_fitness;
  using evolutionary_solver_cpu<TFloat>::_max_agent_idx;

  using evolutionary_solver_cpu<TFloat>::_min_agent_genome;
  using evolutionary_solver_cpu<TFloat>::_min_agent_fitness;
  using evolutionary_solver_cpu<TFloat>::_min_agent_idx;

  using evolutionary_solver_cpu<TFloat>::_migration_step;
  using evolutionary_solver_cpu<TFloat>::_migration_size;
  using evolutionary_solver_cpu<TFloat>::_migration_selection_size;
  using evolutionary_solver_cpu<TFloat>::_migration_idxs;
  using evolutionary_solver_cpu<TFloat>::_migration_buffer;

  using evolutionary_solver_cpu<TFloat>::_bulk_prn_generator;
  using evolutionary_solver_cpu<TFloat>::_bulk_prns;
  using evolutionary_solver_cpu<TFloat>::_bulk_size;
  using evolutionary_solver_cpu<TFloat>::_prn_sets;

  using evolutionary_solver_cpu<TFloat>::_generation_count;
  using evolutionary_solver_cpu<TFloat>::_generation_target;
  using evolutionary_solver_cpu<TFloat>::_f_initialized;
};

} // namespace locusta
#include "ga_solver_cpu_impl.hpp"
#endif
