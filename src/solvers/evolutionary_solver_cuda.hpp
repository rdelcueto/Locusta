#ifndef LOCUSTA_EVOLUTIONARY_SOLVER_CUDA_H
#define LOCUSTA_EVOLUTIONARY_SOLVER_CUDA_H

#include <iostream>
#include <limits>

#include "cuda_common/cuda_helpers.h"

#include "../evaluator/evaluator_cuda.hpp"
#include "../population/population_set_cuda.hpp"
#include "../prngenerator/prngenerator_cuda.hpp"

#include "evolutionary_solver.hpp"

namespace locusta {

/**
 * @brief CUDA implementation of the evolutionary_solver class.
 *
 * This class extends the evolutionary_solver class with CUDA-specific
 * functionality.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct evolutionary_solver_cuda : evolutionary_solver<TFloat>
{

  /**
   * @brief Construct a new evolutionary_solver_cuda object.
   *
   * @param population Population set.
   * @param evaluator Evaluator.
   * @param prn_generator Pseudo-random number generator.
   * @param generation_target Target number of generations.
   * @param upper_bounds Array of upper bounds for the genes.
   * @param lower_bounds Array of lower bounds for the genes.
   */
  evolutionary_solver_cuda(population_set_cuda<TFloat>* population,
                           evaluator_cuda<TFloat>* evaluator,
                           prngenerator_cuda<TFloat>* prn_generator,
                           uint64_t generation_target,
                           TFloat* upper_bounds,
                           TFloat* lower_bounds);

  /**
   * @brief Destroy the evolutionary_solver_cuda object.
   */
  virtual ~evolutionary_solver_cuda();

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
  virtual void teardown_solver() = 0;

  /**
   * @brief Apply the solver's population transformation.
   *
   * This method applies the solver's specific population transformation to
   * generate the next generation of candidate solutions.
   */
  virtual void transform() = 0;

  /**
   * @brief Evaluate the genomes.
   *
   * This method calls the evaluator and assigns a fitness value to every genome
   * in the population.
   */
  virtual void evaluate_genomes();

  /**
   * @brief Update the best genomes records.
   *
   * This method updates the records of the best genomes found so far.
   */
  virtual void update_records();

  /**
   * @brief Regenerate the pseudo-random numbers.
   *
   * This method regenerates the bulk pseudo-random numbers used by the solver.
   */
  virtual void regenerate_prns();

  /**
   * @brief Crop a vector to fit within the bounds.
   *
   * This method crops the values of a vector to fit within the solver's bounds.
   *
   * @param vec Vector to crop.
   */
  virtual void crop_vector(TFloat* vec);

  /**
   * @brief Initialize a vector with uniform random values within the bounds.
   *
   * This method initializes a vector with uniform random values within the
   * solver's bounds.
   *
   * @param dst_vec Vector to initialize.
   */
  virtual void initialize_vector(TFloat* dst_vec);

  /**
   * @brief Print the current population.
   *
   * This method prints all current genomes and their fitness.
   */
  virtual void print_population();

  /**
   * @brief Print the last transformation difference.
   *
   * This method prints the difference between the current population and the
   * previous population after the last transformation.
   */
  virtual void print_transformation_diff();

  /**
   * @brief Print the solver's current best found solutions.
   *
   * This method prints the solver's current best found solutions and their
   * fitness.
   */
  virtual void print_solutions();

  /// Specialized device pointers
  evaluator_cuda<TFloat>* _dev_evaluator;
  population_set_cuda<TFloat>* _dev_population;
  prngenerator_cuda<TFloat>* _dev_bulk_prn_generator;

  /// Genes Variable Bounds (DEVICE COPY)
  TFloat* _DEV_UPPER_BOUNDS;
  TFloat* _DEV_LOWER_BOUNDS;

  /// Variable Ranges (DEVICE COPY)
  TFloat* _DEV_VAR_RANGES;

  /// Stores the agent's genome, which has max fitness, per
  /// isle. (DEVICE COPY)
  TFloat* _dev_max_agent_genome;
  /// Stores the agent's genome, which has min fitness, per
  /// isle. (DEVICE COPY)
  TFloat* _dev_min_agent_genome;

  /// Stores the agent's fitness, which has max fitness, per
  /// isle. (DEVICE COPY)
  TFloat* _dev_max_agent_fitness;
  /// Stores the agent's fitness, which has min fitness, per
  /// isle. (DEVICE COPY)
  TFloat* _dev_min_agent_fitness;

  /// Stores the agent's index, which has max fitness, per isle. (DEVICE
  /// COPY)
  uint32_t* _dev_max_agent_idx;
  /// Stores the agent's index, which has min fitness, per isle. (DEVICE
  /// COPY)
  uint32_t* _dev_min_agent_idx;

  /// Describes the migration selection indexes. (DEVICE COPY)
  uint32_t* _dev_migration_idxs;

  /// Stores the temporal migration genomes to be migrated.
  TFloat* _dev_migration_buffer;

  /// Bulk Pseudo Random Number array
  TFloat* _dev_bulk_prns;

  /// Populations Configuration
  using evolutionary_solver<TFloat>::_ISLES;
  using evolutionary_solver<TFloat>::_AGENTS;
  using evolutionary_solver<TFloat>::_DIMENSIONS;

  /// Genes Variable Bounds (HOST COPY)
  using evolutionary_solver<TFloat>::_UPPER_BOUNDS;
  using evolutionary_solver<TFloat>::_LOWER_BOUNDS;
  /// Variable Ranges (HOST COPY)
  using evolutionary_solver<TFloat>::_VAR_RANGES;

  using evolutionary_solver<TFloat>::_evaluator;
  using evolutionary_solver<TFloat>::_population;

  using evolutionary_solver<TFloat>::_max_agent_genome;
  using evolutionary_solver<TFloat>::_min_agent_genome;
  using evolutionary_solver<TFloat>::_max_agent_fitness;
  using evolutionary_solver<TFloat>::_min_agent_fitness;
  using evolutionary_solver<TFloat>::_max_agent_idx;
  using evolutionary_solver<TFloat>::_min_agent_idx;

  using evolutionary_solver<TFloat>::_migration_step;
  using evolutionary_solver<TFloat>::_migration_size;
  using evolutionary_solver<TFloat>::_migration_selection_size;
  /// Describes the migration selection indexes. (HOST COPY)
  using evolutionary_solver<TFloat>::_migration_idxs;
  /// Stores the temporal migration genomes to be migrated. (HOST COPY)
  using evolutionary_solver<TFloat>::_migration_buffer;

  using evolutionary_solver<TFloat>::_bulk_prn_generator;
  using evolutionary_solver<TFloat>::_bulk_size;

  using evolutionary_solver<TFloat>::_generation_count;
  using evolutionary_solver<TFloat>::_generation_target;
  using evolutionary_solver<TFloat>::_f_initialized;
};

} // namespace locusta
#include "evolutionary_solver_cuda_impl.hpp"
#endif
