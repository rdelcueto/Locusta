#ifndef LOCUSTA_DE_SOLVER_CUDA_H
#define LOCUSTA_DE_SOLVER_CUDA_H

#include "../../prngenerator/prngenerator_cuda.hpp"
#include "../evolutionary_solver_cuda.hpp"

#include "./de_operators/de_operators_cuda.hpp"

namespace locusta {

/**
 * @brief CUDA implementation of the differential evolution solver.
 *
 * This class implements the differential evolution solver for the CUDA
 * architecture.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct de_solver_cuda : evolutionary_solver_cuda<TFloat>
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
   * @brief Construct a new de_solver_cuda object.
   *
   * @param population Population set.
   * @param evaluator Evaluator.
   * @param prn_generator Pseudo-random number generator.
   * @param generation_target Target number of generations.
   * @param upper_bounds Array of upper bounds for the genes.
   * @param lower_bounds Array of lower bounds for the genes.
   */
  de_solver_cuda(population_set_cuda<TFloat>* population,
                 evaluator_cuda<TFloat>* evaluator,
                 prngenerator_cuda<TFloat>* prn_generator,
                 uint64_t generation_target,
                 TFloat* upper_bounds,
                 TFloat* lower_bounds);

  /**
   * @brief Destroy the de_solver_cuda object.
   */
  virtual ~de_solver_cuda();

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
   * @brief Advance the solver by one generation step.
   *
   * This method evolves the population through one generation step.
   */
  virtual void advance();

  /**
   * @brief Apply the solver's population transformation.
   *
   * This method applies the solver's specific population transformation to
   * generate the next generation of candidate solutions.
   */
  virtual void transform();

  /**
   * @brief Replace the trial vector.
   *
   * This method replaces the trial vector with the best candidate solution.
   */
  virtual void trial_vector_replace();

  /**
   * @brief Set the differential evolution solver operators.
   *
   * This method sets the differential evolution solver operators, including the
   * breeding and selection operators.
   *
   * @param breed_functor_ptr Breeding operator.
   * @param select_functor_ptr Selection operator.
   */
  virtual void setup_operators(
    DeBreedCudaFunctor<TFloat>* breed_functor_ptr,
    DeSelectionCudaFunctor<TFloat>* select_functor_ptr);

  /**
   * @brief Configure the solver.
   *
   * This method sets up the solver's configuration, including parameters for
   * migration, selection, crossover, and differential scale factor.
   *
   * @param migration_step Migration step size.
   * @param migration_size Migration size.
   * @param migration_selection_size Migration selection size.
   * @param selection_size Selection size.
   * @param selection_stochastic_factor Selection stochastic factor.
   * @param crossover_rate Crossover rate.
   * @param differential_scale_factor Differential scale factor.
   */
  virtual void solver_config(uint32_t migration_step,
                             uint32_t migration_size,
                             uint32_t migration_selection_size,
                             uint32_t selection_size,
                             TFloat selection_stochastic_factor,
                             TFloat crossover_rate,
                             TFloat differential_scale_factor);

  /// Population crossover operator.
  DeBreedCudaFunctor<TFloat>* _breed_functor_ptr;

  /// Population couple selection.
  DeSelectionCudaFunctor<TFloat>* _selection_functor_ptr;

  /// Tournament selection size.
  uint32_t _selection_size;

  /// Tournament stochastic factor
  TFloat _selection_stochastic_factor;

  /// Crossover rate.
  TFloat _crossover_rate;

  /// Differential scale factor.
  TFloat _differential_scale_factor;

  /// Describes the best position's fitness per particle.
  TFloat* _dev_previous_fitness_array;

  /// Trial vector selection array.
  uint32_t* _dev_recombination_idx_array;

  /// Temporal candidate selection array
  uint32_t* _dev_recombination_reservoir_array;

  // CUDA specific Evolutionary solver vars
  using evolutionary_solver_cuda<TFloat>::_dev_population;
  using evolutionary_solver_cuda<TFloat>::_dev_evaluator;
  using evolutionary_solver_cuda<TFloat>::_dev_bulk_prn_generator;

  using evolutionary_solver_cuda<TFloat>::_DEV_UPPER_BOUNDS;
  using evolutionary_solver_cuda<TFloat>::_DEV_LOWER_BOUNDS;
  using evolutionary_solver_cuda<TFloat>::_DEV_VAR_RANGES;

  using evolutionary_solver_cuda<TFloat>::_dev_max_agent_genome;
  using evolutionary_solver_cuda<TFloat>::_dev_max_agent_fitness;
  using evolutionary_solver_cuda<TFloat>::_dev_max_agent_idx;

  using evolutionary_solver_cuda<TFloat>::_dev_min_agent_genome;
  using evolutionary_solver_cuda<TFloat>::_dev_min_agent_fitness;
  using evolutionary_solver_cuda<TFloat>::_dev_min_agent_idx;

  using evolutionary_solver_cuda<TFloat>::_dev_migration_idxs;
  using evolutionary_solver_cuda<TFloat>::_dev_migration_buffer;
  using evolutionary_solver_cuda<TFloat>::_dev_bulk_prns;

  // Evolutionary solver vars
  using evolutionary_solver_cuda<TFloat>::_ISLES;
  using evolutionary_solver_cuda<TFloat>::_AGENTS;
  using evolutionary_solver_cuda<TFloat>::_DIMENSIONS;

  using evolutionary_solver_cuda<TFloat>::_UPPER_BOUNDS;
  using evolutionary_solver_cuda<TFloat>::_LOWER_BOUNDS;
  using evolutionary_solver_cuda<TFloat>::_VAR_RANGES;

  using evolutionary_solver_cuda<TFloat>::_population;
  using evolutionary_solver_cuda<TFloat>::_evaluator;

  using evolutionary_solver_cuda<TFloat>::_max_agent_genome;
  using evolutionary_solver_cuda<TFloat>::_max_agent_fitness;
  using evolutionary_solver_cuda<TFloat>::_max_agent_idx;

  using evolutionary_solver_cuda<TFloat>::_min_agent_genome;
  using evolutionary_solver_cuda<TFloat>::_min_agent_fitness;
  using evolutionary_solver_cuda<TFloat>::_min_agent_idx;

  using evolutionary_solver_cuda<TFloat>::_migration_step;
  using evolutionary_solver_cuda<TFloat>::_migration_size;
  using evolutionary_solver_cuda<TFloat>::_migration_selection_size;
  using evolutionary_solver_cuda<TFloat>::_migration_idxs;
  using evolutionary_solver_cuda<TFloat>::_migration_buffer;

  using evolutionary_solver_cuda<TFloat>::_bulk_prn_generator;
  using evolutionary_solver_cuda<TFloat>::_bulk_prns;
  using evolutionary_solver_cuda<TFloat>::_bulk_size;
  using evolutionary_solver_cuda<TFloat>::_prn_sets;

  using evolutionary_solver_cuda<TFloat>::_generation_count;
  using evolutionary_solver_cuda<TFloat>::_generation_target;
  using evolutionary_solver_cuda<TFloat>::_f_initialized;
};

} // namespace locusta
#include "de_solver_cuda_impl.hpp"
#endif
