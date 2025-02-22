#include "ga_solver_cpu.hpp"

namespace locusta {

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
template<typename TFloat>
ga_solver_cpu<TFloat>::ga_solver_cpu(population_set_cpu<TFloat>* population,
                                     evaluator_cpu<TFloat>* evaluator,
                                     prngenerator_cpu<TFloat>* prn_generator,
                                     uint64_t generation_target,
                                     TFloat* upper_bounds,
                                     TFloat* lower_bounds)

  : evolutionary_solver_cpu<TFloat>(population,
                                    evaluator,
                                    prn_generator,
                                    generation_target,
                                    upper_bounds,
                                    lower_bounds)
{
  // Defaults
  _migration_step = 0;
  _migration_size = 1;
  _migration_selection_size = 2;
  _selection_size = 2;
  _selection_stochastic_factor = 0;
  _crossover_rate = 0.9;
  _mutation_rate = 0.1;

  // Allocate GA resources
  const size_t TOTAL_GENES = _population->_TOTAL_GENES;
  const size_t TOTAL_AGENTS = _population->_TOTAL_AGENTS;

  _couples_idx_array = new uint32_t[TOTAL_AGENTS];
}

/**
 * @brief Destroy the ga_solver_cpu object.
 */
template<typename TFloat>
ga_solver_cpu<TFloat>::~ga_solver_cpu()
{
  delete[] _couples_idx_array;
}

/**
 * @brief Set up the solver.
 *
 * This method initializes and allocates the solver's runtime resources.
 */
template<typename TFloat>
void
ga_solver_cpu<TFloat>::setup_solver()
{
  // Pseudo random number allocation.
  const uint32_t SELECTION_OFFSET = _selection_functor_ptr->required_prns(this);
  const uint32_t BREEDING_OFFSET = _breed_functor_ptr->required_prns(this);

  _bulk_size = SELECTION_OFFSET + BREEDING_OFFSET;
  _bulk_prns = new TFloat[_bulk_size];

  _prn_sets = new TFloat*[2];
  _prn_sets[SELECTION_SET] = _bulk_prns;
  _prn_sets[BREEDING_SET] = _bulk_prns + SELECTION_OFFSET;

  evolutionary_solver_cpu<TFloat>::setup_solver();
}

/**
 * @brief Tear down the solver.
 *
 * This method terminates and deallocates the solver's runtime resources.
 */
template<typename TFloat>
void
ga_solver_cpu<TFloat>::teardown_solver()
{
  delete[] _prn_sets;
  delete[] _bulk_prns;
}

/**
 * @brief Set the genetic algorithm solver operators.
 *
 * This method sets the genetic algorithm solver operators, including the
 * breeding and selection operators.
 *
 * @param breed_functor_ptr Breeding operator.
 * @param selection_functor_ptr Selection operator.
 */
template<typename TFloat>
void
ga_solver_cpu<TFloat>::setup_operators(
  GaBreedFunctor<TFloat>* breed_functor_ptr,
  GaSelectionFunctor<TFloat>* selection_functor_ptr)
{
  _breed_functor_ptr = breed_functor_ptr;
  _selection_functor_ptr = selection_functor_ptr;
}

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
template<typename TFloat>
void
ga_solver_cpu<TFloat>::solver_config(uint32_t migration_step,
                                     uint32_t migration_size,
                                     uint32_t migration_selection_size,
                                     uint32_t selection_size,
                                     TFloat selection_stochastic_factor,
                                     TFloat crossover_rate,
                                     TFloat mutation_rate)
{
  _migration_step = migration_step;
  _migration_size = migration_size;
  _migration_selection_size = migration_selection_size;
  _selection_size = selection_size;
  _selection_stochastic_factor = selection_stochastic_factor;
  _crossover_rate = crossover_rate;
  _mutation_rate = mutation_rate;
}

/**
 * @brief Apply the solver's population transformation.
 *
 * This method applies the solver's specific population transformation to
 * generate the next generation of candidate solutions.
 */
template<typename TFloat>
void
ga_solver_cpu<TFloat>::transform()
{
  elite_population_replace();

  (*_selection_functor_ptr)(this);
  (*_breed_functor_ptr)(this);

  // Crop transformation vector
  // evolutionary_solver_cpu<TFloat>::crop_vector(
  //     _population->_transformed_data_array);
}

/**
 * @brief Replace the elite population.
 *
 * This method replaces the elite population with the best individuals from the
 * current population.
 */
template<typename TFloat>
void
ga_solver_cpu<TFloat>::elite_population_replace()
{
  TFloat* genomes = _population->_data_array;
  TFloat* fitness = _population->_fitness_array;

  // Scan population
  for (uint32_t i = 0; i < _ISLES; i++) {
    const uint32_t min_idx = _min_agent_idx[i];

    fitness[i * _AGENTS + min_idx] = _max_agent_fitness[i];

    const TFloat* max_genome = _max_agent_genome + i * _DIMENSIONS;
    TFloat* min_genome =
      genomes + i * _AGENTS * _DIMENSIONS + min_idx * _DIMENSIONS;

    for (uint32_t k = 0; k < _DIMENSIONS; k++) {
      min_genome[k] = max_genome[k];
    }
  }
}

} // namespace locusta
