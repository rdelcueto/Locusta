#ifndef LOCUSTA_EVOLUTIONARY_SOLVER_CPU_H
#define LOCUSTA_EVOLUTIONARY_SOLVER_CPU_H

#include <iostream>
#include <limits>

#include "../evaluator/evaluator_cpu.hpp"
#include "../population/population_set_cpu.hpp"
#include "../prngenerator/prngenerator_cpu.hpp"

#include "evolutionary_solver.hpp"

namespace locusta {

/**
 * @brief CPU implementation of the evolutionary_solver class.
 *
 * This class extends the evolutionary_solver class with CPU-specific
 * functionality.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct evolutionary_solver_cpu : evolutionary_solver<TFloat>
{

  /**
   * @brief Construct a new evolutionary_solver_cpu object.
   *
   * @param population Population set.
   * @param evaluator Evaluator.
   * @param prn_generator Pseudo-random number generator.
   * @param generation_target Target number of generations.
   * @param upper_bounds Array of upper bounds for the genes.
   * @param lower_bounds Array of lower bounds for the genes.
   */
  evolutionary_solver_cpu(population_set_cpu<TFloat>* population,
                          evaluator_cpu<TFloat>* evaluator,
                          prngenerator_cpu<TFloat>* prn_generator,
                          uint64_t generation_target,
                          TFloat* upper_bounds,
                          TFloat* lower_bounds);

  /**
   * @brief Destroy the evolutionary_solver_cpu object.
   */
  virtual ~evolutionary_solver_cpu();

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
  virtual void transform() = 0;

  /**
   * @brief Update the best genomes records.
   *
   * This method updates the records of the best genomes found so far.
   */
  virtual void update_records();

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
