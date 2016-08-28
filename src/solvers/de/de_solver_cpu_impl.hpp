#include "de_solver_cpu.hpp"

namespace locusta {

/// Interface for Differential Evolution solvers
template <typename TFloat>
de_solver_cpu<TFloat>::de_solver_cpu(population_set_cpu<TFloat>* population,
                                     evaluator_cpu<TFloat>* evaluator,
                                     prngenerator_cpu<TFloat>* prn_generator,
                                     uint32_t generation_target,
                                     TFloat* upper_bounds, TFloat* lower_bounds)

  : evolutionary_solver_cpu<TFloat>(population, evaluator, prn_generator,
                                    generation_target, upper_bounds,
                                    lower_bounds)
{
  // Defaults
  _migration_step = 0;
  _migration_size = 1;
  _migration_selection_size = 2;
  _selection_size = 2;
  _selection_stochastic_factor = 0;
  _crossover_rate = 0.9;
  _differential_scale_factor = 0.5;

  // Allocate GA resources
  const size_t TOTAL_AGENTS = _population->_TOTAL_AGENTS;
  const size_t RANDOM_VECTORS = 3;

  _previous_fitness_array = new TFloat[TOTAL_AGENTS];
  _recombination_idx_array = new uint32_t[TOTAL_AGENTS * RANDOM_VECTORS];
}

template <typename TFloat>
de_solver_cpu<TFloat>::~de_solver_cpu()
{
  delete[] _previous_fitness_array;
  delete[] _recombination_idx_array;
}

template <typename TFloat>
void
de_solver_cpu<TFloat>::setup_solver()
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

  evolutionary_solver_cpu<TFloat>::evaluate_genomes();
}

template <typename TFloat>
void
de_solver_cpu<TFloat>::teardown_solver()
{
  delete[] _prn_sets;
  delete[] _bulk_prns;
}

template <typename TFloat>
void
de_solver_cpu<TFloat>::setup_operators(
  DeBreedFunctor<TFloat>* breed_functor_ptr,
  DeSelectionFunctor<TFloat>* selection_functor_ptr)
{
  _breed_functor_ptr = breed_functor_ptr;
  _selection_functor_ptr = selection_functor_ptr;
}

template <typename TFloat>
void
de_solver_cpu<TFloat>::solver_config(uint32_t migration_step,
                                     uint32_t migration_size,
                                     uint32_t migration_selection_size,
                                     uint32_t selection_size,
                                     TFloat selection_stochastic_factor,
                                     TFloat crossover_rate,
                                     TFloat differential_scale_factor)
{
  _migration_step = migration_step;
  _migration_size = migration_size;
  _migration_selection_size = migration_selection_size;
  _selection_size = selection_size;
  _selection_stochastic_factor = selection_stochastic_factor;
  _crossover_rate = crossover_rate;
  _differential_scale_factor = differential_scale_factor;
}

template <typename TFloat>
void
de_solver_cpu<TFloat>::advance()
{
  // Store previous fitness evaluation values.
  const size_t TOTAL_AGENTS = _population->_TOTAL_AGENTS;
  TFloat* population_data_fitness = _population->_fitness_array;
  memcpy(_previous_fitness_array, population_data_fitness,
         TOTAL_AGENTS * sizeof(TFloat));

  transform();

  _population->swap_data_sets();
  // Evaluate trial vectors
  evolutionary_solver_cpu<TFloat>::evaluate_genomes();
  // Replace target vectors with trial vectors, if better solutions found
  trial_vector_replace();
  // Restore target vectors
  _population->swap_data_sets();

  evolutionary_solver_cpu<TFloat>::update_records();
  evolutionary_solver_cpu<TFloat>::regenerate_prns();

  _generation_count++;
}

template <typename TFloat>
void
de_solver_cpu<TFloat>::transform()
{

  (*_selection_functor_ptr)(this);
  (*_breed_functor_ptr)(this);
}

template <typename TFloat>
void
de_solver_cpu<TFloat>::trial_vector_replace()
{
  TFloat* previous_vectors = _population->_transformed_data_array;
  TFloat* previous_fitness = _previous_fitness_array;

  TFloat* trial_vectors = _population->_data_array;
  TFloat* trial_fitness = _population->_fitness_array;

  // Scan population
  for (uint32_t i = 0; i < _ISLES; i++) {
    for (uint32_t j = 0; j < _AGENTS; j++) {
      const uint32_t FITNESS_OFFSET = i * _AGENTS + j;
      if (trial_fitness[FITNESS_OFFSET] > previous_fitness[FITNESS_OFFSET]) {
        // Replace target vector with trial vector
        const uint32_t GENOME_OFFSET =
          i * _AGENTS * _DIMENSIONS + j * _DIMENSIONS;
        const TFloat* trial_vector = trial_vectors + GENOME_OFFSET;
        TFloat* target_vector = previous_vectors + GENOME_OFFSET;

        for (uint32_t k = 0; k < _DIMENSIONS; k++) {
          target_vector[k] = trial_vector[k];
        }
      } else {
        // Keep target value fitness
        trial_fitness[FITNESS_OFFSET] = previous_fitness[FITNESS_OFFSET];
      }
    }
  }
}

} // namespace locusta
