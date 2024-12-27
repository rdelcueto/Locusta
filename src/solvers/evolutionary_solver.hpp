#ifndef LOCUSTA_EVOLUTIONARY_SOLVER_H
#define LOCUSTA_EVOLUTIONARY_SOLVER_H

#include <limits>

#include "../evaluator/evaluator.hpp"
#include "../population/population_set.hpp"
#include "../prngenerator/prngenerator.hpp"

#include <iostream>

namespace locusta {

/**
 * @brief Interface for evolutionary computing metaheuristic solvers.
 *
 * This abstract class defines the interface for evolutionary computing
 * metaheuristic solvers. Concrete solver implementations, such as the genetic
 * algorithm, particle swarm optimization, and differential evolution solvers,
 * derive from this abstract class.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct evolutionary_solver
{

  /**
   * @brief Construct a new evolutionary_solver object.
   *
   * @param population Population set.
   * @param evaluator Evaluator.
   * @param prn_generator Pseudo-random number generator.
   * @param generation_target Target number of generations.
   * @param upper_bounds Array of upper bounds for the genes.
   * @param lower_bounds Array of lower bounds for the genes.
   */
  evolutionary_solver(population_set<TFloat>* population,
                      evaluator<TFloat>* evaluator,
                      prngenerator<TFloat>* prn_generator,
                      uint64_t generation_target,
                      TFloat* upper_bounds,
                      TFloat* lower_bounds);

  /**
   * @brief Destroy the evolutionary_solver object.
   */
  virtual ~evolutionary_solver();

  /**
   * @brief Set up the solver.
   *
   * This method initializes and allocates the solver's runtime resources.
   */
  virtual void setup_solver() = 0;

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
   * @brief Advance the solver by one generation step.
   *
   * This method evolves the population through one generation step.
   */
  virtual void advance();

  /**
   * @brief Run the solver.
   *
   * This method runs the solver until it reaches the target number of
   * generations.
   */
  virtual void run();

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
  virtual void update_records() = 0;

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
  virtual void crop_vector(TFloat* vec) = 0;

  /**
   * @brief Initialize a vector with uniform random values within the bounds.
   *
   * This method initializes a vector with uniform random values within the
   * solver's bounds.
   *
   * @param dst_vec Vector to initialize.
   */
  virtual void initialize_vector(TFloat* dst_vec) = 0;

  /**
   * @brief Print the current population.
   *
   * This method prints all current genomes and their fitness.
   */
  virtual void print_population() = 0;

  /**
   * @brief Print the last transformation difference.
   *
   * This method prints the difference between the current population and the
   * previous population after the last transformation.
   */
  virtual void print_transformation_diff() = 0;

  /**
   * @brief Print the solver's current best found solutions.
   *
   * This method prints the solver's current best found solutions and their
   * fitness.
   */
  virtual void print_solutions() = 0;

  /// Evaluator
  evaluator<TFloat>* const _evaluator;

  /// Bulk Pseudo Random Number Generator
  prngenerator<TFloat>* const _bulk_prn_generator;

  /// Population Set
  population_set<TFloat>* const _population;

  /// Populations Configuration
  const uint32_t _ISLES;
  const uint32_t _AGENTS;
  const uint32_t _DIMENSIONS;

  /// Genes Variable Bounds
  TFloat* _UPPER_BOUNDS;
  TFloat* _LOWER_BOUNDS;

  /// Variable Ranges
  TFloat* _VAR_RANGES;

  /// Stores the agent's genome, which has max fitness, per isle.
  TFloat* _max_agent_genome;

  /// Stores the agent's genome, which has min fitness, per isle.
  TFloat* _min_agent_genome;

  /// Stores the agent's fitness, which has max fitness, per isle.
  TFloat* _max_agent_fitness;

  /// Stores the agent's fitness, which has min fitness, per isle.
  TFloat* _min_agent_fitness;

  /// Stores the agent's index, which has max fitness, per isle.
  uint32_t* _max_agent_idx;

  /// Stores the agent's index, which has min fitness, per isle.
  uint32_t* _min_agent_idx;

  /// Defines the migration size.
  uint32_t _migration_step;

  /// Defines the migration size.
  uint32_t _migration_size;

  /// Defines the migration selection window size.
  uint32_t _migration_selection_size;

  /// Describes the migration selection indexes.
  uint32_t* _migration_idxs;

  /// Stores the temporal migration genomes to be migrated.
  TFloat* _migration_buffer;

  /// Bulk Pseudo Random Number array
  TFloat* _bulk_prns;

  /// Describes the size of the _bulk_prnumbers array.
  uint32_t _bulk_size;

  /// Describes the locations of each pseudo random number set.
  TFloat** _prn_sets;

  /// Counter describing the solver's current generation.
  uint32_t _generation_count;

  /// Defines the solver's target generation.
  uint64_t _generation_target;

  uint8_t _f_initialized;
};

} // namespace locusta
#include "evolutionary_solver_impl.hpp"
#endif
